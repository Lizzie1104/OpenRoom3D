#!/usr/bin/env python3
"""Local upload/train/query application. Uses the frozen openroom3d environment."""
from __future__ import annotations
import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid

from flask import Flask, jsonify, request, render_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORE = ROOT / "outputs/application"
STORE.mkdir(parents=True, exist_ok=True)
from query_negatives import DEFAULT_NEGATIVES, without_query
app = Flask(__name__, template_folder=str(ROOT / "openroom3d/web"))
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 ** 3
jobs = {}
lock = threading.Lock()
gpu_lock = threading.Lock()
viewers = {}
for record in STORE.glob('*/job.json'):
    try:
        old=json.loads(record.read_text())
        if old.get('status') in ('queued','running'):
            old.update(status='interrupted',error='The application restarted. Intermediate files are available. Submit the job again.')
            record.write_text(json.dumps(old,ensure_ascii=False,indent=2))
        if old['id'] == record.parent.name and len(old['id']) == 32 and all(c in '0123456789abcdef' for c in old['id']):
            jobs[old['id']]=old
    except (ValueError,KeyError,OSError):
        pass


def configs():
    return {str(p.relative_to(ROOT)): p for p in sorted((ROOT / "outputs").glob("*/feature-splatting/*/config.yml"))
            if list((p.parent / "nerfstudio_models").glob("*.ckpt"))}


def save(job):
    (STORE / job['id'] / "job.json").write_text(json.dumps({k:v for k,v in job.items() if k != 'process'}, ensure_ascii=False, indent=2))


def create_job(kind):
    key = uuid.uuid4().hex
    (STORE / key).mkdir()
    job = dict(id=key, kind=kind, status="queued", stage="Waiting for compute", error=None)
    jobs[key] = job
    save(job)
    return job


def command(job, argv, stage):
    if job['status'] == 'cancelled':
        raise RuntimeError('Job cancelled')
    job.update(status="running", stage=stage)
    save(job)
    with (STORE / job['id'] / "run.log").open('a') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        job['process'] = proc
        code = proc.wait()
        job.pop('process', None)
        if code:
            raise RuntimeError(f'{stage} failed (exit code {code}). See the job output.')


def background(job, work):
    def run():
        try:
            with gpu_lock:
                if job['status'] == 'cancelled': return
                work()
                job.update(status='complete', stage='Complete')
        except Exception as error:
            if job['status'] != 'cancelled': job.update(status='failed', error=str(error))
        finally:
            save(job)
    threading.Thread(target=run, daemon=True).start()


@app.before_request
def local_request():
    if request.host.split(':')[0].lower() not in ('localhost', '127.0.0.1'):
        return jsonify(error='Unrecognized host.'), 403
    if request.method != 'GET':
        origin = request.headers.get('Origin')
        if request.headers.get('Sec-Fetch-Site') == 'cross-site' or (origin and origin != request.host_url.rstrip('/')):
            return jsonify(error='Cross-origin request rejected.'), 403


@app.get('/')
def index(): return render_template('index.html', default_negatives=DEFAULT_NEGATIVES)


@app.get('/api/scenes')
def scenes(): return jsonify(list(configs()))


@app.post('/api/upload')
def upload():
    video = request.files.get('video')
    if not video or Path(video.filename or '').suffix.lower() not in ('.mp4', '.mov', '.mkv', '.avi'):
        return jsonify(error='Select an MP4, MOV, MKV or AVI video.'), 400
    job = create_job('train')
    if jobs and any(j['id']!=job['id'] and j['status'] in ('queued','running') for j in jobs.values()):
        job.update(status='cancelled',error='Finish or cancel the active job first.');save(job)
        return jsonify(error='A compute job is active. Finish or cancel it before uploading another video.'),409
    name = 'scene_' + job['id'][:12]
    path = STORE / job['id'] / ('input' + Path(video.filename).suffix.lower())
    video.save(path)
    data = ROOT / 'data/processed' / name
    def work():
        command(job, [sys.executable, '-m', 'nerfstudio.scripts.process_data', 'video', '--data', str(path), '--output-dir', str(data)], 'Extracting frames and estimating cameras with COLMAP')
        # Feature Splatting jointly trains geometry and distilled features;
        # no separate Splatfacto training is required before this stage.
        command(job, [sys.executable, '-m', 'nerfstudio.scripts.train', 'feature-splatting', '--data', str(data),
                      '--experiment-name', name, '--output-dir', str(ROOT/'outputs'), '--max-num-iterations', '30000', '--vis', 'tensorboard'], 'Training geometry and semantic features (30,000 steps)')
        available = [p for k,p in configs().items() if p.parent.parent.parent.name == name]
        if not available: raise RuntimeError('Training finished without a valid checkpoint.')
        from trusted_scene import register_training
        selected = sorted(available)[-1]
        register_training(selected, STORE / job['id'])
        job['config'] = selected.relative_to(ROOT).as_posix()
    background(job, work)
    return jsonify(id=job['id'])


@app.post('/api/query')
def query():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict) or not isinstance(body.get('config'), str) or not isinstance(body.get('query'), str):
        return jsonify(error='Expected a scene name and a text query.'), 400
    config = configs().get(body.get('config'))
    text = str(body.get('query','')).strip()
    if not config or not text or len(text)>200 or any(ord(c)<32 for c in text):
        return jsonify(error='Select a trained scene and enter a query.'),400
    try:
        threshold=float(body.get('threshold',.5)); eps=float(body.get('eps',.02))
        if not 0<threshold<1 or not 0<eps<=1: raise ValueError()
    except (TypeError,ValueError): return jsonify(error='Set threshold in (0, 1) and cluster radius in (0, 1].'),400
    negative=body.get('negative') or DEFAULT_NEGATIVES
    if not isinstance(negative,str) or len(negative)>1000 or any(ord(c)<32 for c in negative):
        return jsonify(error='Negatives must be comma-separated text.'),400
    try: without_query(negative,text)
    except ValueError: return jsonify(error='Every negative matches the query. Enter other negatives.'),400
    job=create_job('query')
    result=STORE/job['id']/'result'
    def work():
        command(job,[sys.executable,str(ROOT/'scripts/prepare_query_object.py'),'--config',str(config),'--query',text,
                     '--output',str(result),'--negative',negative,'--threshold',str(threshold),'--eps',str(eps)],'Scoring text, refining geometry and estimating poses')
        manifest=json.loads((result/'manifest.json').read_text())
        job['candidates']=[dict(label=r['viewer_label'],count=r['num_gaussians'],cluster=r['cluster_label']) for r in manifest['candidates'] if r['viewer_label']]
        job['query']=text
    background(job,work)
    return jsonify(id=job['id'])


@app.get('/api/jobs/<key>')
def status(key):
    job=jobs.get(key)
    if not job: return jsonify(error='Job not found'),404
    log=STORE/key/'run.log'
    payload={k:v for k,v in job.items() if k!='process'}
    payload['log']=log.read_text(errors='replace')[-12000:] if log.exists() else ''
    return jsonify(payload)


@app.post('/api/jobs/<key>/cancel')
def cancel(key):
    job=jobs.get(key)
    if not job:return jsonify(error='Job not found'),404
    if job['status'] in ('queued','running'):
        job['status']='cancelled'
        proc=job.get('process')
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        save(job)
    return jsonify(status=job['status'])


@app.post('/api/jobs/<key>/viewer')
def viewer(key):
    job=jobs.get(key)
    if not job or job['kind']!='query' or job['status']!='complete':return jsonify(error='Complete a query first.'),400
    manifest=json.loads((STORE/key/'result/manifest.json').read_text())
    candidates=[r for r in manifest['candidates'] if r['viewer_label']]
    if not candidates:return jsonify(error='No candidates. Lower the threshold or try another query.'),400
    with lock:
        for proc,log in viewers.values():
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        viewers.clear()
        argv=[sys.executable,str(ROOT/'scripts/object_pose_viewer.py'),'--scene',manifest['scene_npz'],'--port','7020']
        for row in candidates:argv.extend(['--object',row['viewer_label']+'='+row['npz']])
        log=(STORE/key/'viewer.log').open('a')
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        viewers[key]=(proc,log)
    return jsonify(url='http://localhost:7020',note='Starting the viewer. Disable the heatmap to move a candidate.')


@app.get('/api/jobs/<key>/viewer-status')
def viewer_status(key):
    entry=viewers.get(key)
    if not entry:return jsonify(error='Viewer has not started.'),404
    proc,_=entry
    log=STORE/key/'viewer.log'
    return jsonify(running=proc.poll() is None, log=log.read_text(errors='replace')[-4000:])


def shutdown():
    processes=[j.get('process') for j in jobs.values()]+[v[0] for v in viewers.values()]
    for proc in processes:
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass


if __name__=='__main__':
    atexit.register(shutdown)
    print('OpenRoom3D application at http://localhost:7000',flush=True)
    app.run(host='127.0.0.1',port=7000,debug=False,threaded=True)
