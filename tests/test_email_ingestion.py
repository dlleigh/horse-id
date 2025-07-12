import pytest
import os
import re
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import pandas as pd
from dateutil import tz

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest_from_email import (
    gmail_authenticate,
    get_emails_to_process,
    extract_horse_name,
    extract_oldest_date
)


class TestGmailAuthentication:
    """Test Gmail authentication functionality."""
    
    @patch('ingest_from_email.os.path.exists')
    @patch('ingest_from_email.Credentials.from_authorized_user_file')
    @patch('ingest_from_email.build')
    def test_gmail_authenticate_with_valid_token(self, mock_build, mock_from_file, mock_exists):
        """Test authentication with valid existing token."""
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = True
        mock_from_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        result = gmail_authenticate()
        
        assert result == mock_service
        mock_from_file.assert_called_once()
        mock_build.assert_called_once_with('gmail', 'v1', credentials=mock_creds)
    
    @patch('ingest_from_email.os.path.exists')
    @patch('ingest_from_email.Credentials.from_authorized_user_file')
    @patch('ingest_from_email.Request')
    @patch('ingest_from_email.build')
    def test_gmail_authenticate_with_expired_token(self, mock_build, mock_request, mock_from_file, mock_exists):
        """Test authentication with expired token that can be refreshed."""
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = 'refresh_token'
        mock_from_file.return_value = mock_creds
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        with patch('ingest_from_email.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = gmail_authenticate()
        
        assert result == mock_service
        mock_creds.refresh.assert_called_once()
        mock_build.assert_called_once_with('gmail', 'v1', credentials=mock_creds)
    
    @patch('ingest_from_email.os.path.exists')
    @patch('ingest_from_email.Credentials.from_authorized_user_file')
    @patch('ingest_from_email.InstalledAppFlow.from_client_secrets_file')
    @patch('ingest_from_email.build')
    def test_gmail_authenticate_new_auth_flow(self, mock_build, mock_flow, mock_from_file, mock_exists):
        """Test authentication with new authorization flow."""
        mock_exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = False
        mock_from_file.return_value = mock_creds
        
        mock_flow_instance = Mock()
        mock_new_creds = Mock()
        mock_flow_instance.run_local_server.return_value = mock_new_creds
        mock_flow.return_value = mock_flow_instance
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        with patch('ingest_from_email.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = gmail_authenticate()
        
        assert result == mock_service
        mock_flow.assert_called_once()
        mock_flow_instance.run_local_server.assert_called_once_with(port=0)
        mock_build.assert_called_once_with('gmail', 'v1', credentials=mock_new_creds)
    
    @patch('ingest_from_email.os.path.exists')
    @patch('ingest_from_email.InstalledAppFlow.from_client_secrets_file')
    @patch('ingest_from_email.build')
    def test_gmail_authenticate_no_existing_token(self, mock_build, mock_flow, mock_exists):
        """Test authentication with no existing token."""
        mock_exists.return_value = False
        
        mock_flow_instance = Mock()
        mock_new_creds = Mock()
        mock_flow_instance.run_local_server.return_value = mock_new_creds
        mock_flow.return_value = mock_flow_instance
        
        mock_service = Mock()
        mock_build.return_value = mock_service
        
        with patch('ingest_from_email.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = gmail_authenticate()
        
        assert result == mock_service
        mock_flow.assert_called_once()
        mock_flow_instance.run_local_server.assert_called_once_with(port=0)
        mock_build.assert_called_once_with('gmail', 'v1', credentials=mock_new_creds)


class TestGetEmailsToProcess:
    """Test email retrieval functionality."""
    
    def test_get_emails_to_process_single_page(self):
        """Test retrieving emails with single page of results."""
        mock_service = Mock()
        mock_messages = Mock()
        mock_list = Mock()
        mock_service.users.return_value.messages.return_value.list.return_value = mock_list
        
        mock_result = {
            'messages': [
                {'id': 'msg1'},
                {'id': 'msg2'},
                {'id': 'msg3'}
            ]
        }
        mock_list.execute.return_value = mock_result
        
        existing_message_ids = {'msg1'}
        
        result = get_emails_to_process(mock_service, existing_message_ids)
        
        assert len(result) == 2  # msg2 and msg3 (msg1 is excluded)
        assert result[0]['id'] == 'msg2'
        assert result[1]['id'] == 'msg3'
    
    def test_get_emails_to_process_multiple_pages(self):
        """Test retrieving emails with multiple pages of results."""
        mock_service = Mock()
        mock_messages = Mock()
        mock_list = Mock()
        mock_service.users.return_value.messages.return_value.list.return_value = mock_list
        
        # First page
        mock_result1 = {
            'messages': [{'id': 'msg1'}, {'id': 'msg2'}],
            'nextPageToken': 'token123'
        }
        # Second page
        mock_result2 = {
            'messages': [{'id': 'msg3'}, {'id': 'msg4'}]
        }
        
        mock_list.execute.side_effect = [mock_result1, mock_result2]
        
        existing_message_ids = {'msg1'}
        
        result = get_emails_to_process(mock_service, existing_message_ids)
        
        assert len(result) == 3  # msg2, msg3, msg4 (msg1 is excluded)
        assert mock_list.execute.call_count == 2
    
    def test_get_emails_to_process_no_messages(self):
        """Test retrieving emails with no messages."""
        mock_service = Mock()
        mock_messages = Mock()
        mock_list = Mock()
        mock_service.users.return_value.messages.return_value.list.return_value = mock_list
        
        mock_result = {}  # No messages key
        mock_list.execute.return_value = mock_result
        
        existing_message_ids = set()
        
        result = get_emails_to_process(mock_service, existing_message_ids)
        
        assert len(result) == 0
    
    def test_get_emails_to_process_api_error(self):
        """Test retrieving emails with API error."""
        mock_service = Mock()
        mock_messages = Mock()
        mock_list = Mock()
        mock_service.users.return_value.messages.return_value.list.return_value = mock_list
        
        mock_list.execute.side_effect = Exception('API error')
        
        existing_message_ids = set()
        
        result = get_emails_to_process(mock_service, existing_message_ids)
        
        assert len(result) == 0
    
    def test_get_emails_to_process_all_existing(self):
        """Test retrieving emails where all messages already exist."""
        mock_service = Mock()
        mock_messages = Mock()
        mock_list = Mock()
        mock_service.users.return_value.messages.return_value.list.return_value = mock_list
        
        mock_result = {
            'messages': [
                {'id': 'msg1'},
                {'id': 'msg2'},
                {'id': 'msg3'}
            ]
        }
        mock_list.execute.return_value = mock_result
        
        existing_message_ids = {'msg1', 'msg2', 'msg3'}
        
        result = get_emails_to_process(mock_service, existing_message_ids)
        
        assert len(result) == 0


class TestExtractHorseName:
    """Test horse name extraction functionality."""
    
    def test_extract_horse_name_basic(self):
        """Test basic horse name extraction."""
        subject = "Thunder-fall 2023"
        result = extract_horse_name(subject)
        assert result == "Thunder"
    
    def test_extract_horse_name_with_spaces(self):
        """Test horse name extraction with spaces."""
        subject = "Sky Blue-spring photos"
        result = extract_horse_name(subject)
        assert result == "Sky Blue"
    
    def test_extract_horse_name_with_forwarded_prefix(self):
        """Test horse name extraction with forwarded email prefix."""
        subject = "Fwd: Lightning-summer 2023"
        result = extract_horse_name(subject)
        assert result == "Lightning"
    
    def test_extract_horse_name_case_insensitive_seasons(self):
        """Test horse name extraction with different season cases."""
        test_cases = [
            ("Thunder-FALL 2023", "Thunder"),
            ("Lightning-Winter photos", "Lightning"),
            ("Storm-SUMMER session", "Storm"),
            ("Breeze-spring update", "Breeze")
        ]
        
        for subject, expected in test_cases:
            result = extract_horse_name(subject)
            assert result == expected
    
    def test_extract_horse_name_with_special_characters(self):
        """Test horse name extraction with special characters removal."""
        subject = "Thunder's Boy!-fall 2023"
        result = extract_horse_name(subject)
        assert result == "Thunders Boy"
    
    def test_extract_horse_name_no_season_pattern(self):
        """Test horse name extraction with no season pattern."""
        subject = "Random email subject"
        result = extract_horse_name(subject)
        assert result is None
    
    def test_extract_horse_name_empty_subject(self):
        """Test horse name extraction with empty subject."""
        subject = ""
        result = extract_horse_name(subject)
        assert result is None
    
    def test_extract_horse_name_multiple_dashes(self):
        """Test horse name extraction with multiple dashes."""
        subject = "Thunder Lightning-fall 2023"
        result = extract_horse_name(subject)
        assert result == "Thunder Lightning"  # Takes everything before dash + season
    
    def test_extract_horse_name_no_dash_before_season(self):
        """Test horse name extraction without dash before season."""
        subject = "Thunder fall 2023"
        result = extract_horse_name(subject)
        assert result == "Thunder"
    
    def test_extract_horse_name_with_numbers(self):
        """Test horse name extraction with numbers in name."""
        subject = "Thunder2-fall 2023"
        result = extract_horse_name(subject)
        assert result == "Thunder2"


class TestExtractOldestDate:
    """Test oldest date extraction functionality."""
    
    def test_extract_oldest_date_from_headers(self):
        """Test date extraction from email headers."""
        message = {
            'internalDate': '1672592400000',  # Jan 1, 2023 12:00:00 UTC in milliseconds
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Wed, 01 Jan 2023 12:00:00 -0500'},
                    {'name': 'Subject', 'value': 'Test'}
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        assert result == '20230101'  # Should return YYYYMMDD format
    
    def test_extract_oldest_date_multiple_headers(self):
        """Test date extraction with multiple date headers."""
        message = {
            'internalDate': '1672592400000',  # Jan 1, 2023 12:00:00 UTC in milliseconds  
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Wed, 01 Jan 2023 12:00:00 -0500'},
                    {'name': 'Received', 'value': 'from server; Tue, 31 Dec 2022 10:00:00 -0500'}
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should return the date in YYYYMMDD format
        assert result == '20230101'
    
    def test_extract_oldest_date_from_body(self):
        """Test date extraction from email body."""
        import base64
        
        body_text = "From: user@example.com\nDate: Mon, 15 May 2023 08:30:00 -0400\nSubject: Test"
        encoded_body = base64.urlsafe_b64encode(body_text.encode('utf-8')).decode('utf-8')
        
        message = {
            'internalDate': '1672592400000',  # Jan 1, 2023 12:00:00 UTC in milliseconds
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Wed, 01 Jan 2023 12:00:00 -0500'}
                ],
                'parts': [
                    {
                        'mimeType': 'text/plain',
                        'body': {
                            'data': encoded_body
                        }
                    }
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should return date in YYYYMMDD format
        assert result == '20230101'
    
    def test_extract_oldest_date_timezone_handling(self):
        """Test date extraction with different timezones."""
        message = {
            'internalDate': '1672592400000',  # Jan 1, 2023 12:00:00 UTC in milliseconds
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Wed, 01 Jan 2023 12:00:00 PST'},
                    {'name': 'Received', 'value': 'from server; Wed, 01 Jan 2023 15:00:00 EST'}
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should return date in YYYYMMDD format
        assert result == '20230101'
    
    def test_extract_oldest_date_no_dates(self):
        """Test date extraction with no valid dates."""
        from datetime import datetime
        message = {
            'internalDate': '0',  # Invalid timestamp
            'payload': {
                'headers': [
                    {'name': 'Subject', 'value': 'Test email'},
                    {'name': 'From', 'value': 'user@example.com'}
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should return today's date as fallback
        today = datetime.now().date().strftime('%Y%m%d')
        assert result == today
    
    def test_extract_oldest_date_invalid_format(self):
        """Test date extraction with invalid date format."""
        from datetime import datetime
        message = {
            'internalDate': '0',  # Invalid timestamp
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Invalid date format'},
                    {'name': 'Subject', 'value': 'Test'}
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should return today's date as fallback
        today = datetime.now().date().strftime('%Y%m%d')
        assert result == today
    
    def test_extract_oldest_date_nested_parts(self):
        """Test date extraction with nested message parts."""
        import base64
        
        body_text = "Original message:\nDate: Fri, 10 Feb 2023 14:00:00 -0600"
        encoded_body = base64.urlsafe_b64encode(body_text.encode('utf-8')).decode('utf-8')
        
        message = {
            'internalDate': '1672592400000',  # Jan 1, 2023 12:00:00 UTC in milliseconds
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Wed, 01 Jan 2023 12:00:00 -0500'}
                ],
                'parts': [
                    {
                        'mimeType': 'multipart/mixed',
                        'parts': [
                            {
                                'mimeType': 'text/plain',
                                'body': {
                                    'data': encoded_body
                                }
                            }
                        ]
                    }
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should return date in YYYYMMDD format
        assert result == '20230101'
    
    def test_extract_oldest_date_body_decode_error(self):
        """Test date extraction with body decode error."""
        message = {
            'internalDate': '1672592400000',  # Jan 1, 2023 12:00:00 UTC in milliseconds
            'payload': {
                'headers': [
                    {'name': 'Date', 'value': 'Wed, 01 Jan 2023 12:00:00 -0500'}
                ],
                'parts': [
                    {
                        'mimeType': 'text/plain',
                        'body': {
                            'data': 'invalid_base64_data'
                        }
                    }
                ]
            }
        }
        
        result = extract_oldest_date(message)
        
        # Should still return the date in YYYYMMDD format
        assert result == '20230101'