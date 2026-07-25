# Model storage options

The worker has two independently versioned inputs: the executable runtime and
the three reviewed model snapshots. They do not have to share one OCI image,
but both must remain immutable and independently verifiable.

## Options

| Option | New-Pod path | Reproducibility | Operational trade-off |
|---|---|---|---|
| Full immutable image | Pull runtime and all three models as OCI layers | Strongest: one digest names everything | Simplest contract, but every uncached host pulls the largest image |
| Runtime image plus prepared Network Volume | Pull runtime image, mount an already verified model bundle | Strong when both image digest and bundle digest are required | Smaller image and one model upload per datacenter; volume locality constrains GPU placement |
| Runtime image plus per-Pod model download | Pull runtime, then download three repositories at startup | Possible only with exact revisions and file manifests | Repeats roughly 10 GB of transfer and depends on upstream availability; not a fast-start design |
| Runtime image plus object-storage bundle | Pull runtime, then fetch one digest-addressed bundle | Strong with bundle digest and short-lived access | Portable across datacenters, but repeats the model transfer for every uncached Pod |

For repeated Pods in one Secure Cloud datacenter, the prepared Network Volume
is the only separated-model option likely to improve warm startup. A model
download on every Pod merely moves bytes out of the OCI pull and adds another
failure mode.

## Prepared-volume contract

The model artifact is a versioned, read-only directory such as:

```text
/models-volume/bundles/<bundle-sha256>/
  MODELS.json
  manifests/{whisper,qwen,reazon}.sha256
  models/{whisper,qwen,reazon}/...
  READY.json
```

`READY.json` is written atomically only after every file and the aggregate
bundle digest have been verified. The runtime image accepts an explicit bundle
digest, resolves only that directory, rejects symlinks and unexpected files,
and repeats manifest verification before reporting ready. Model updates create
a new directory; they never mutate a ready bundle in place.

Mount the Network Volume at a model-only path such as `/models-volume`. Keep job
audio, transcripts, tokens, and temporary chunks on the container disk under
`/workspace/multi-asr`; they must not persist in the shared model volume.

## Runpod-specific consequences

- Network Volumes are available only for Secure Cloud Pods and are tied to a
  datacenter. This can reduce the set of available GPUs for a new Pod.
- Network storage performance is variable. Model load may be slower than local
  container storage even when startup transfer is avoided.
- A Pod with a Network Volume cannot be stopped; it must be terminated, while
  the volume remains independently billable and reusable.
- Network Volumes are not encrypted by Runpod. This design stores only public
  model artifacts there, never meeting data or credentials.

## Decision and proof plan

Use the full immutable image as the first measured baseline because it has one
artifact identity and no prerequisite volume. Then compare a runtime-only image
against a `model_volume_ready` volume in the same GPU class and datacenter.
Record these separately:

1. image or runtime pull to worker process start;
2. full model-manifest verification;
3. each backend's model-load and inference time;
4. Pod-create to locally verified three-model result;
5. image bytes, volume bytes, GPU availability, and storage cost.

Adopt the prepared-volume variant only if its warm-volume result is materially
faster and the datacenter constraint is acceptable. Do not claim its one-time
volume population as Pod startup performance; report that cold preparation as a
separate measurement.
