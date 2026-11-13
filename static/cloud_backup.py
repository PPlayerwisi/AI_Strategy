import os
import json
import base64
import threading
import urllib.request
import urllib.error
import uuid

from game_config import (
    POOL_DIR, HUMAN_TELEMETRY_FILE, POOL_REGISTRY_FILE, USERS_DB_FILE
)

BACKUP_LOCK = threading.Lock()

class CloudBackupManager:
    @staticmethod
    def _get_github_config() -> dict:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        repo = os.environ.get("GITHUB_BACKUP_REPO", "").strip()
        branch = os.environ.get("GITHUB_BACKUP_BRANCH", "models-backup").strip() or "models-backup"
        return {
            "enabled": bool(token and repo),
            "token": token,
            "repo": repo,
            "branch": branch
        }

    @staticmethod
    def _github_request(url: str, token: str, method: str = "GET", payload: dict = None):
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "FactionWarfareRTS-Backup",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    @staticmethod
    def _ensure_github_branch(cfg: dict) -> bool:
        repo = cfg["repo"]
        branch = cfg["branch"]
        token = cfg["token"]
        branch_url = f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}"
        try:
            CloudBackupManager._github_request(branch_url, token, method="GET")
            return True
        except urllib.error.HTTPError as e:
            if e.code != 404:
                return False
        except Exception:
            return False

        try:
            repo_info = CloudBackupManager._github_request(f"https://api.github.com/repos/{repo}", token, method="GET")
            default_branch = repo_info.get("default_branch", "main")
            ref_info = CloudBackupManager._github_request(
                f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}",
                token,
                method="GET"
            )
            base_sha = ref_info["object"]["sha"]
            CloudBackupManager._github_request(
                f"https://api.github.com/repos/{repo}/git/refs",
                token,
                method="POST",
                payload={"ref": f"refs/heads/{branch}", "sha": base_sha}
            )
            return True
        except Exception:
            return False

    @staticmethod
    def upload_file_to_github(local_path: str, remote_path: str, commit_msg: str = "Auto-backup AI artifact") -> bool:
        cfg = CloudBackupManager._get_github_config()
        if not cfg["enabled"] or not os.path.exists(local_path):
            return False

        with BACKUP_LOCK:
            if not CloudBackupManager._ensure_github_branch(cfg):
                return False

            repo = cfg["repo"]
            branch = cfg["branch"]
            token = cfg["token"]
            api_url = f"https://api.github.com/repos/{repo}/contents/{remote_path}"

            existing_sha = None
            try:
                info = CloudBackupManager._github_request(f"{api_url}?ref={branch}", token, method="GET")
                existing_sha = info.get("sha")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    return False
            except Exception:
                return False

            try:
                with open(local_path, "rb") as f:
                    b64_content = base64.b64encode(f.read()).decode("ascii")

                payload = {
                    "message": commit_msg,
                    "content": b64_content,
                    "branch": branch
                }
                if existing_sha:
                    payload["sha"] = existing_sha

                CloudBackupManager._github_request(api_url, token, method="PUT", payload=payload)
                return True
            except Exception as ex:
                print(f"[Backup] GitHub upload error for {remote_path}: {ex}")
                return False

    @staticmethod
    def restore_from_github() -> int:
        cfg = CloudBackupManager._get_github_config()
        if not cfg["enabled"]:
            return 0

        os.makedirs(POOL_DIR, exist_ok=True)
        repo = cfg["repo"]
        branch = cfg["branch"]
        token = cfg["token"]
        dir_url = f"https://api.github.com/repos/{repo}/contents/backup_pool?ref={branch}"

        try:
            items = CloudBackupManager._github_request(dir_url, token, method="GET")
            if not isinstance(items, list):
                return 0
        except Exception:
            return 0

        restored_count = 0
        for item in items:
            if item.get("type") != "file":
                continue
            fname = item.get("name", "")
            dl_url = item.get("download_url", "")
            if not fname or not dl_url:
                continue

            local_target = os.path.join(POOL_DIR, fname)
            should_download = (
                not os.path.exists(local_target)
                or fname in ("users_db.json", "pool_registry.json", "human_telemetry.npz")
            )
            if not should_download:
                continue

            try:
                req = urllib.request.Request(
                    dl_url,
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "FactionWarfareRTS-Backup"}
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    content_bytes = resp.read()
                if content_bytes:
                    tmp_path = local_target + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(content_bytes)
                    os.replace(tmp_path, local_target)
                    restored_count += 1
            except Exception as ex:
                print(f"[Backup] GitHub restore error for {fname}: {ex}")

        if restored_count > 0:
            print(f"[Backup] Restored {restored_count} files from GitHub branch '{branch}'.")
        return restored_count

    @staticmethod
    def _post_multipart(url: str, fields: dict, file_field_name: str = None, file_path: str = None):
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()

        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.extend(f"{v}\r\n".encode("utf-8"))

        if file_field_name and file_path and os.path.exists(file_path):
            fname = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                f_bytes = f.read()
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{file_field_name}"; filename="{fname}"\r\n'.encode("utf-8"))
            body.extend(b"Content-Type: application/zip\r\n\r\n")
            body.extend(f_bytes)
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "FactionWarfareRTS-Notifier"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.read()

    @staticmethod
    def send_discord_notification(message_text: str, model_file_path: str = None) -> bool:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        if not webhook_url:
            return False
        try:
            CloudBackupManager._post_multipart(
                webhook_url,
                fields={"content": message_text},
                file_field_name="file",
                file_path=model_file_path
            )
            return True
        except Exception as ex:
            print(f"[Backup] Discord notification error: {ex}")
            return False

    @staticmethod
    def send_telegram_notification(message_text: str, model_file_path: str = None) -> bool:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not bot_token or not chat_id:
            return False
        try:
            if model_file_path and os.path.exists(model_file_path):
                url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
                CloudBackupManager._post_multipart(
                    url,
                    fields={"chat_id": chat_id, "caption": message_text},
                    file_field_name="document",
                    file_path=model_file_path
                )
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                CloudBackupManager._post_multipart(
                    url,
                    fields={"chat_id": chat_id, "text": message_text}
                )
            return True
        except Exception as ex:
            print(f"[Backup] Telegram notification error: {ex}")
            return False

    @staticmethod
    def sync_users_db_async():
        def _worker():
            if os.path.exists(USERS_DB_FILE):
                CloudBackupManager.upload_file_to_github(
                    USERS_DB_FILE,
                    "backup_pool/users_db.json",
                    commit_msg="Sync player ratings and profiles"
                )
        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def backup_new_generation_async(entry: dict, model_zip_path: str):
        def _worker():
            label = entry.get("label", "Gen-Unknown")
            rating = float(entry.get("rating", 0.00))
            gauntlet = float(entry.get("w_gauntlet", 0.00))
            doc = float(entry.get("doctrine_alignment", 100.00))
            origin = entry.get("origin", "player")
            fname = os.path.basename(model_zip_path)

            gh_ok = False
            if os.path.exists(model_zip_path):
                gh_ok = CloudBackupManager.upload_file_to_github(
                    model_zip_path,
                    f"backup_pool/{fname}",
                    commit_msg=f"Backup model {label} (Rating: {rating:.2f})"
                )
            if os.path.exists(POOL_REGISTRY_FILE):
                CloudBackupManager.upload_file_to_github(
                    POOL_REGISTRY_FILE,
                    "backup_pool/pool_registry.json",
                    commit_msg=f"Update registry for {label}"
                )
            if os.path.exists(USERS_DB_FILE):
                CloudBackupManager.upload_file_to_github(
                    USERS_DB_FILE,
                    "backup_pool/users_db.json",
                    commit_msg="Update user profiles"
                )
            if os.path.exists(HUMAN_TELEMETRY_FILE):
                CloudBackupManager.upload_file_to_github(
                    HUMAN_TELEMETRY_FILE,
                    "backup_pool/human_telemetry.npz",
                    commit_msg="Update elite human telemetry"
                )

            gh_status = "Збережено в GitHub" if gh_ok else "GitHub не налаштовано"
            msg = (
                f"**Нова нейромережа еволюціонувала!**\n"
                f"• **Покоління:** `{label}`\n"
                f"• **Рейтинг:** `{rating:.2f}`\n"
                f"• **Gauntlet Winrate:** `{gauntlet:.2f}%`\n"
                f"• **Відповідність доктрині:** `{doc:.2f}%`\n"
                f"• **Джерело:** `{origin}`\n"
                f"• **Бекап:** {gh_status}"
            )
            CloudBackupManager.send_discord_notification(msg, model_file_path=model_zip_path)
            CloudBackupManager.send_telegram_notification(msg.replace("**", ""), model_file_path=model_zip_path)

        threading.Thread(target=_worker, daemon=True).start()