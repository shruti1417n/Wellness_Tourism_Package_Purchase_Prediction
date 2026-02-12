from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("tourism"))
api.upload_folder(
    folder_path="/content/drive/MyDrive/PythonPrograms/MLops_Devops/tourism_project",     # the local folder containing your files
    repo_id="ShrutiHulyal/WellnessPackagePrediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
