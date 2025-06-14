from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os
from typing import List, Dict, Optional
from .config import settings

class GmailClient:
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Gmail API 인증을 수행합니다."""
        if os.path.exists(settings.GMAIL_TOKEN_FILE):
            with open(settings.GMAIL_TOKEN_FILE, 'rb') as token:
                self.creds = pickle.load(token)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.GMAIL_CREDENTIALS_FILE, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open(settings.GMAIL_TOKEN_FILE, 'wb') as token:
                pickle.dump(self.creds, token)
        
        self.service = build('gmail', 'v1', credentials=self.creds)
    
    def get_unread_emails(self, max_results: int = 10) -> List[Dict]:
        """읽지 않은 이메일을 가져옵니다."""
        try:
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['UNREAD'],
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()
                
                headers = msg['payload']['headers']
                subject = next(h['value'] for h in headers if h['name'] == 'Subject')
                sender = next(h['value'] for h in headers if h['name'] == 'From')
                
                # 이메일 본문 추출
                body = ''
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            body = part['body'].get('data', '')
                            break
                elif 'body' in msg['payload']:
                    body = msg['payload']['body'].get('data', '')
                
                emails.append({
                    'id': message['id'],
                    'subject': subject,
                    'sender': sender,
                    'body': body
                })
            
            return emails
        except Exception as e:
            print(f"이메일 가져오기 실패: {str(e)}")
            return []
    
    def send_reply(self, message_id: str, reply_text: str) -> bool:
        """이메일에 답장을 보냅니다."""
        try:
            # 원본 메시지 가져오기
            original = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='metadata',
                metadataHeaders=['Subject', 'From']
            ).execute()
            
            headers = original['payload']['headers']
            subject = next(h['value'] for h in headers if h['name'] == 'Subject')
            to = next(h['value'] for h in headers if h['name'] == 'From')
            
            # 답장 메시지 생성
            message = {
                'raw': self._create_message(to, f"Re: {subject}", reply_text)
            }
            
            # 메시지 전송
            self.service.users().messages().send(
                userId='me',
                body=message
            ).execute()
            
            return True
        except Exception as e:
            print(f"답장 전송 실패: {str(e)}")
            return False
    
    def _create_message(self, to: str, subject: str, message_text: str) -> str:
        """이메일 메시지를 생성합니다."""
        import base64
        from email.mime.text import MIMEText
        
        message = MIMEText(message_text)
        message['to'] = to
        message['subject'] = subject
        
        return base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    def mark_as_read(self, message_id: str) -> bool:
        """이메일을 읽음으로 표시합니다."""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            print(f"읽음 표시 실패: {str(e)}")
            return False
    
    def delete_email(self, message_id: str) -> bool:
        """이메일을 삭제합니다."""
        try:
            self.service.users().messages().delete(
                userId='me',
                id=message_id
            ).execute()
            return True
        except Exception as e:
            print(f"이메일 삭제 실패: {str(e)}")
            return False 