# Trying another object

Use the same feature checkpoint for new text queries in an existing scene. Run the following from the project root, replacing RUN with the directory that contains the desired configuration.

```bash
python scripts/prepare_query_object.py --config "$RUN/config.yml" \
  --query cup --output outputs/cup_trial --port 7013
```

Inspect the generated candidate report before selecting a cluster. The largest cluster can include the support surface, and a small component can be a fragment. Try an absent query as a negative control. The default negatives include desk02 categories, and any negative sharing a word with the query is dropped. Pass `--negative` with categories from the new scene when the scene changes. A thermos-like target is an example where a different description can better match the actual object. Use a new output directory for each threshold or cluster restriction.

The candidate NPZ retains Gaussian indices and original parameters. The pose sidecar records its source and coordinate frame. Use the generated viewer command to compare candidates individually. Absolute orientation about the symmetry axis of a cup is particularly weakly constrained. A new video requires its own camera reconstruction, checkpoint and feature export.
