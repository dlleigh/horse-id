import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from horse_id import horse_id_processor_handler, process_image_for_identification, _parse_twilio_event


class TestTwilioEventParsing:
    """Test Twilio event parsing functionality."""
    
    def test_parse_direct_payload(self):
        """Test parsing direct payload (not from API Gateway)."""
        event = {
            'MediaUrl0': 'http://example.com/image.jpg',
            'From': '+15551234567',
            'To': '+15557654321'
        }
        
        result = _parse_twilio_event(event)
        
        assert result == event
    
    def test_parse_form_urlencoded_body(self):
        """Test parsing form-urlencoded body from API Gateway."""
        event = {
            'body': 'MediaUrl0=http%3A//example.com/image.jpg&From=%2B15551234567&To=%2B15557654321',
            'headers': {'content-type': 'application/x-www-form-urlencoded'},
            'isBase64Encoded': False
        }
        
        result = _parse_twilio_event(event)
        
        assert result['MediaUrl0'] == 'http://example.com/image.jpg'
        assert result['From'] == '+15551234567'
        assert result['To'] == '+15557654321'
    
    def test_parse_json_body(self):
        """Test parsing JSON body from API Gateway."""
        payload = {
            'MediaUrl0': 'http://example.com/image.jpg',
            'From': '+15551234567',
            'To': '+15557654321'
        }
        event = {
            'body': json.dumps(payload),
            'headers': {'content-type': 'application/json'},
            'isBase64Encoded': False
        }
        
        result = _parse_twilio_event(event)
        
        assert result == payload
    
    def test_parse_base64_encoded_body(self):
        """Test parsing base64 encoded body."""
        import base64
        
        original_body = 'MediaUrl0=http%3A//example.com/image.jpg&From=%2B15551234567'
        encoded_body = base64.b64encode(original_body.encode('utf-8')).decode('utf-8')
        
        event = {
            'body': encoded_body,
            'headers': {'content-type': 'application/x-www-form-urlencoded'},
            'isBase64Encoded': True
        }
        
        result = _parse_twilio_event(event)
        
        assert result['MediaUrl0'] == 'http://example.com/image.jpg'
        assert result['From'] == '+15551234567'
    
    def test_parse_invalid_base64_body(self):
        """Test parsing invalid base64 body."""
        event = {
            'body': 'invalid_base64_content',
            'headers': {'content-type': 'application/x-www-form-urlencoded'},
            'isBase64Encoded': True
        }
        
        result = _parse_twilio_event(event)
        
        assert result == {}
    
    def test_parse_empty_body(self):
        """Test parsing empty body."""
        event = {
            'body': '',
            'headers': {'content-type': 'application/x-www-form-urlencoded'},
            'isBase64Encoded': False
        }
        
        result = _parse_twilio_event(event)
        
        assert result == {}
    
    def test_parse_unsupported_content_type(self):
        """Test parsing unsupported content type."""
        event = {
            'body': 'some content',
            'headers': {'content-type': 'text/plain'},
            'isBase64Encoded': False
        }
        
        result = _parse_twilio_event(event)
        
        assert result == {}


class TestHorseIdProcessorHandler:
    """Test horse ID processor handler functionality."""
    
    def test_missing_image_url(self):
        """Test handler with missing image URL."""
        event = {
            'body': 'From=%2B15551234567&To=%2B15557654321&Body=test',
            'headers': {'content-type': 'application/x-www-form-urlencoded'}
        }
        context = {}
        
        response = horse_id_processor_handler(event, context)
        
        assert response['statusCode'] == 400
        assert 'No image URL found' in response['body']
    
    @patch('horse_id.process_image_for_identification')
    def test_successful_processing(self, mock_process_image):
        """Test successful image processing."""
        mock_process_image.return_value = {
            'status': 'success',
            'predictions': [
                {'identity': 'Thunder', 'score': 0.85},
                {'identity': 'Lightning', 'score': 0.72}
            ]
        }
        
        event = {
            'body': 'MediaUrl0=http%3A//example.com/image.jpg&From=%2B15551234567&To=%2B15557654321',
            'headers': {'content-type': 'application/x-www-form-urlencoded'}
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_ACCOUNT_SID': 'test_sid',
            'TWILIO_AUTH_TOKEN': 'test_token'
        }, clear=True):
            with patch('horse_id.Client') as mock_twilio_client:
                mock_message = Mock()
                mock_message.sid = 'test_message_sid'
                mock_twilio_client.return_value.messages.create.return_value = mock_message
                
                response = horse_id_processor_handler(event, context)
        
        assert response['statusCode'] == 200
        assert 'test_message_sid' in response['body']
        
        # Verify image processing was called
        mock_process_image.assert_called_once_with(
            'http://example.com/image.jpg',
            twilio_account_sid='test_sid',
            twilio_auth_token='test_token'
        )
        
        # Verify Twilio message was sent
        mock_twilio_client.return_value.messages.create.assert_called_once()
        call_args = mock_twilio_client.return_value.messages.create.call_args
        assert 'Thunder' in call_args.kwargs['body']
        assert 'Lightning' in call_args.kwargs['body']
        assert '85.0%' in call_args.kwargs['body']
        assert '72.0%' in call_args.kwargs['body']
    
    def test_missing_twilio_credentials(self):
        """Test handler with missing Twilio credentials."""
        event = {
            'body': 'MediaUrl0=http%3A//example.com/image.jpg&From=%2B15551234567&To=%2B15557654321',
            'headers': {'content-type': 'application/x-www-form-urlencoded'}
        }
        context = {}
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('horse_id.load_config') as mock_load_config:
                mock_load_config.return_value = {'twilio': {}}
                
                response = horse_id_processor_handler(event, context)
        
        assert response['statusCode'] == 500
        assert 'Twilio credentials' in response['body']
    
    @patch('horse_id.process_image_for_identification')
    def test_processing_exception(self, mock_process_image):
        """Test handler with processing exception."""
        mock_process_image.side_effect = Exception('Processing failed')
        
        event = {
            'body': 'MediaUrl0=http%3A//example.com/image.jpg&From=%2B15551234567&To=%2B15557654321',
            'headers': {'content-type': 'application/x-www-form-urlencoded'}
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_ACCOUNT_SID': 'test_sid',
            'TWILIO_AUTH_TOKEN': 'test_token'
        }, clear=True):
            response = horse_id_processor_handler(event, context)
        
        assert response['statusCode'] == 500
        assert 'Processing failed' in response['body']
    
    def test_missing_to_from_numbers(self):
        """Test handler with missing To/From numbers."""
        event = {
            'body': 'MediaUrl0=http%3A//example.com/image.jpg',
            'headers': {'content-type': 'application/x-www-form-urlencoded'}
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_ACCOUNT_SID': 'test_sid',
            'TWILIO_AUTH_TOKEN': 'test_token'
        }, clear=True):
            with patch('horse_id.process_image_for_identification') as mock_process_image:
                mock_process_image.return_value = {
                    'status': 'success',
                    'predictions': [{'identity': 'Thunder', 'score': 0.85}]
                }
                
                response = horse_id_processor_handler(event, context)
        
        assert response['statusCode'] == 500
        assert 'To/From numbers not found' in response['body']


class TestProcessImageForIdentification:
    """Test image processing for identification."""
    
    @patch('horse_id.load_config')
    def test_missing_config_file(self, mock_load_config):
        """Test processing with missing config file."""
        mock_load_config.side_effect = FileNotFoundError('Config not found')
        
        with pytest.raises(FileNotFoundError):
            process_image_for_identification('http://example.com/image.jpg')
    
    @patch('horse_id.load_config')
    @patch('horse_id.setup_paths')
    def test_config_key_error(self, mock_setup_paths, mock_load_config):
        """Test processing with missing config keys."""
        mock_load_config.return_value = {}
        mock_setup_paths.side_effect = ValueError('Missing path configuration: Missing key')
        
        with pytest.raises(ValueError):
            process_image_for_identification('http://example.com/image.jpg')
    
    @patch('horse_id.load_config')
    @patch('horse_id.setup_paths')
    @patch('horse_id.boto3.client')
    @patch('horse_id.download_from_s3')
    def test_s3_download_failure(self, mock_download, mock_boto3, mock_setup_paths, mock_load_config):
        """Test processing with S3 download failure."""
        mock_load_config.return_value = {'similarity': {'inference_threshold': 0.6}}
        mock_setup_paths.return_value = ('/tmp/manifest.csv', '/tmp/features', 'test-bucket')
        mock_download.return_value = False
        
        with pytest.raises(RuntimeError, match='Could not retrieve manifest file'):
            process_image_for_identification('http://example.com/image.jpg')
    
    @patch('horse_id.load_config')
    @patch('horse_id.setup_paths')
    @patch('horse_id.boto3.client')
    @patch('horse_id.download_from_s3')
    @patch('horse_id.os.path.isfile')
    def test_missing_manifest_file(self, mock_isfile, mock_download, mock_boto3, mock_setup_paths, mock_load_config):
        """Test processing with missing manifest file."""
        mock_load_config.return_value = {'similarity': {'inference_threshold': 0.6}}
        mock_setup_paths.return_value = ('/tmp/manifest.csv', '/tmp/features', 'test-bucket')
        mock_download.return_value = True
        mock_isfile.return_value = False
        
        with pytest.raises(FileNotFoundError, match='MANIFEST_FILE'):
            process_image_for_identification('http://example.com/image.jpg')
    
    @patch('horse_id.load_config')
    @patch('horse_id.setup_paths')
    @patch('horse_id.boto3.client')
    @patch('horse_id.download_from_s3')
    @patch('horse_id.os.path.isfile')
    @patch('horse_id.os.path.isdir')
    def test_missing_features_dir(self, mock_isdir, mock_isfile, mock_download, mock_boto3, mock_setup_paths, mock_load_config):
        """Test processing with missing features directory."""
        mock_load_config.return_value = {'similarity': {'inference_threshold': 0.6}}
        mock_setup_paths.return_value = ('/tmp/manifest.csv', '/tmp/features', 'test-bucket')
        mock_download.return_value = True
        mock_isfile.return_value = True
        mock_isdir.return_value = False
        
        with pytest.raises(NotADirectoryError, match='FEATURES_DIR'):
            process_image_for_identification('http://example.com/image.jpg')
    
    @patch('horse_id.requests.get')
    def test_image_download_failure(self, mock_get):
        """Test processing with image download failure."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException('Download failed')
        
        with patch('horse_id.load_config') as mock_load_config:
            mock_load_config.return_value = {'similarity': {'inference_threshold': 0.6}}
            
            with patch('horse_id.setup_paths') as mock_setup_paths:
                mock_setup_paths.return_value = ('/tmp/manifest.csv', '/tmp/features', 'test-bucket')
                
                with patch('horse_id.boto3.client'):
                    with patch('horse_id.download_from_s3', return_value=True):
                        with patch('horse_id.os.path.isfile', return_value=True):
                            with patch('horse_id.os.path.isdir', return_value=True):
                                with patch('horse_id.Horses') as mock_horses:
                                    mock_horses.return_value.create_catalogue.return_value = Mock(empty=False)
                                    
                                    with pytest.raises(RuntimeError, match='Error downloading image'):
                                        process_image_for_identification('http://example.com/image.jpg')
    
    @patch('horse_id.requests.get')
    def test_invalid_image_content(self, mock_get):
        """Test processing with invalid image content."""
        mock_response = Mock()
        mock_response.content = b'invalid_image_data'
        mock_get.return_value = mock_response
        
        with patch('horse_id.load_config') as mock_load_config:
            mock_load_config.return_value = {'similarity': {'inference_threshold': 0.6}}
            
            with patch('horse_id.setup_paths') as mock_setup_paths:
                mock_setup_paths.return_value = ('/tmp/manifest.csv', '/tmp/features', 'test-bucket')
                
                with patch('horse_id.boto3.client'):
                    with patch('horse_id.download_from_s3', return_value=True):
                        with patch('horse_id.os.path.isfile', return_value=True):
                            with patch('horse_id.os.path.isdir', return_value=True):
                                with patch('horse_id.Horses') as mock_horses:
                                    mock_horses.return_value.create_catalogue.return_value = Mock(empty=False)
                                    
                                    with pytest.raises(ValueError, match='not a valid image'):
                                        process_image_for_identification('http://example.com/image.jpg')
    
    def test_empty_horse_catalogue(self):
        """Test processing with empty horse catalogue."""
        with patch('horse_id.load_config') as mock_load_config:
            mock_load_config.return_value = {'similarity': {'inference_threshold': 0.6}}
            
            with patch('horse_id.setup_paths') as mock_setup_paths:
                mock_setup_paths.return_value = ('/tmp/manifest.csv', '/tmp/features', 'test-bucket')
                
                with patch('horse_id.boto3.client'):
                    with patch('horse_id.download_from_s3', return_value=True):
                        with patch('horse_id.os.path.isfile', return_value=True):
                            with patch('horse_id.os.path.isdir', return_value=True):
                                with patch('horse_id.Horses') as mock_horses:
                                    mock_horses.return_value.create_catalogue.return_value = Mock(empty=True)
                                    
                                    with pytest.raises(ValueError, match='horse catalogue is empty'):
                                        process_image_for_identification('http://example.com/image.jpg')