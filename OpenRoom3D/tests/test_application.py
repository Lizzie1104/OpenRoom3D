"""App routing validation without uploading or launching expensive training."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import openroom_app


class AppTests(unittest.TestCase):
    def setUp(self): self.client=openroom_app.app.test_client()

    def test_home_and_scene_registry(self):
        self.assertEqual(self.client.get('/').status_code,200)
        with patch.object(openroom_app,'configs',return_value={'scene.yml':Path('/tmp/scene.yml')}):
            self.assertEqual(self.client.get('/api/scenes').json,['scene.yml'])

    def test_reject_invalid_inputs_before_subprocess(self):
        with patch.object(openroom_app,'background') as work:
            self.assertEqual(self.client.post('/api/upload').status_code,400)
            self.assertEqual(self.client.post('/api/query',json={'config':'/etc/passwd','query':'cup'}).status_code,400)
            with patch.object(openroom_app,'configs',return_value={'scene.yml':Path('/tmp/scene.yml')}):
                for values in [{'query':''},{'query':'cup','threshold':2},{'query':'cup','eps':-1},
                               {'query':'table','negative':'table'},{'query':'cup','negative':'a\nb'},{'query':'cup','negative':7}]:
                    values['config']='scene.yml'
                    self.assertEqual(self.client.post('/api/query',json=values).status_code,400)
            work.assert_not_called()

    def test_home_shows_editable_default_negatives(self):
        page=self.client.get('/').get_data(as_text=True)
        self.assertIn('id="negative"',page)
        self.assertIn(openroom_app.DEFAULT_NEGATIVES,page)

    def test_unknown_jobs_and_unfinished_viewer(self):
        self.assertEqual(self.client.get('/api/jobs/not-a-job').status_code,404)
        self.assertEqual(self.client.post('/api/jobs/not-a-job/cancel').status_code,404)
        self.assertEqual(self.client.post('/api/jobs/not-a-job/viewer').status_code,400)
