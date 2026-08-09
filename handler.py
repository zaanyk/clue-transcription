import runpod
from whisper_engine import transcribe_job


def handler(job):
    job_input = job["input"]
    return transcribe_job(job_input)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
