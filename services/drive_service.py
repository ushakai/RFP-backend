"""
Google Drive Service
"""
import io
import traceback
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


def get_drive_service(access_token: str):
    """Create Google Drive service with access token"""
    credentials = Credentials(token=access_token)
    return build('drive', 'v3', credentials=credentials)


def find_or_create_folder(service, folder_name: str, parent_folder_id: str = None) -> str:
    """Find existing folder or create new one"""
    try:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        else:
            query += " and 'root' in parents"
        
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if files:
            return files[0]['id']
        
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
        
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id')
        
    except Exception as e:
        print(f"Error finding/creating folder: {e}")
        traceback.print_exc()
        return None


def upload_file_to_drive(service, file_content: bytes, filename: str, folder_id: str, mime_type: str = 'application/octet-stream') -> str:
    """Upload file to Google Drive folder"""
    try:
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype=mime_type, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
        
    except Exception as e:
        print(f"Error uploading file to Drive: {e}")
        traceback.print_exc()
        return None


def setup_drive_folders(access_token: str, client_name: str) -> dict:
    """Setup folder structure in Google Drive"""
    try:
        service = get_drive_service(access_token)
        
        client_folder_id = find_or_create_folder(service, f"Your_RFP_{client_name}")
        if not client_folder_id:
            return None
        
        processed_folder_id = find_or_create_folder(service, "Processed Files", client_folder_id)
        unprocessed_folder_id = find_or_create_folder(service, "Unprocessed Files", client_folder_id)
        
        return {
            "client_folder_id": client_folder_id,
            "processed_folder_id": processed_folder_id,
            "unprocessed_folder_id": unprocessed_folder_id
        }
        
    except Exception as e:
        print(f"Error setting up Drive folders: {e}")
        traceback.print_exc()
        return None

