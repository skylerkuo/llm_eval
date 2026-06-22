from datetime import datetime
from huggingface_hub import HfApi

repo_id = "Kuoskyler/swiftplan-isaac-sim"

api = HfApi()
info = api.dataset_info(repo_id)

print("Checked time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Dataset:", repo_id)
print("Downloads last 30 days:", info.downloads)

# 有些 huggingface_hub 版本可能沒有 downloads_all_time 屬性，所以用 getattr 比較安全
print("Downloads all time:", getattr(info, "downloads_all_time", "Not available"))