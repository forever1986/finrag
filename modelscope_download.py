from modelscope import snapshot_download
model_dir = snapshot_download(model_id="TongyiFinance/Tongyi-Finance-14B-Chat", cache_dir="./model", ignore_file_pattern=["*.bin"])
