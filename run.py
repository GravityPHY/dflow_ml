from dflow import upload_artifact, download_artifact
from dflow import InputArtifact, OutputArtifact

# upload to container
artifacts_local = upload_artifact("my_project")

from dflow import ShellOPTemplate

test = ShellOPTemplate(
    name="test",
    image="yuh/dflow:0.0.2",
    script="python3 ./project/main.py")
test.inputs.artifacts = {"start": InputArtifact(path="/data/dflow/")}
test.outputs.artifacts = {"end": OutputArtifact(path="result.npz")}
test.outputs.artifacts = {"fig": OutputArtifact(path="pred-test.jpg")}
from dflow import Step


step1 = Step(
    name="step1",
    template=test,
    artifacts={"start": artifacts_local}

)

from dflow import Workflow

# the default host is https://localhost:2746
wf_local = Workflow(name='test')
wf_local.add(step1)
wf_local.submit()