import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webhook_responder import webhook_handler


class TestWebhookHandler:
    """Test webhook handler functionality."""
    
    def test_missing_twilio_auth_token(self):
        """Test handler with missing Twilio auth token."""
        event = {'headers': {}, 'body': ''}
        context = {}
        
        with patch.dict(os.environ, {}, clear=True):
            response = webhook_handler(event, context)
        
        assert response['statusCode'] == 500
        assert 'Content-Type' in response['headers']
        assert response['headers']['Content-Type'] == 'text/xml'
        assert 'Error: Server configuration issue' in response['body']
    
    def test_missing_processor_lambda_name(self):
        """Test handler with missing processor lambda name."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'test_signature'},
            'body': 'test=body',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {'TWILIO_AUTH_TOKEN': 'test_token'}, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                response = webhook_handler(event, context)
        
        assert response['statusCode'] == 500
        assert 'System is not configured correctly' in response['body']
    
    def test_invalid_twilio_signature(self):
        """Test handler with invalid Twilio signature."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'invalid_signature'},
            'body': 'test=body',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {'TWILIO_AUTH_TOKEN': 'test_token'}, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = False
                response = webhook_handler(event, context)
        
        assert response['statusCode'] == 403
        assert 'Forbidden: Invalid Twilio Signature' in response['body']
    
    def test_successful_webhook_processing(self):
        """Test successful webhook processing."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': 'From=%2B15551234567&To=%2B15557654321&Body=test&MediaUrl0=http://example.com/image.jpg',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        assert response['statusCode'] == 200
        assert response['headers']['Content-Type'] == 'text/xml'
        assert 'Identification started!' in response['body']
        assert 'Response' in response['body']
        assert 'Message' in response['body']
        
        # Verify Lambda invocation
        mock_lambda_client.invoke.assert_called_once_with(
            FunctionName='test-processor',
            InvocationType='Event',
            Payload=json.dumps(event)
        )
    
    def test_base64_encoded_body(self):
        """Test handling of base64 encoded body."""
        import base64
        
        original_body = 'From=%2B15551234567&To=%2B15557654321&Body=test'
        encoded_body = base64.b64encode(original_body.encode('utf-8')).decode('utf-8')
        
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': encoded_body,
            'isBase64Encoded': True,
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        assert response['statusCode'] == 200
        mock_lambda_client.invoke.assert_called_once()
    
    def test_invalid_base64_body(self):
        """Test handling of invalid base64 encoded body."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': 'invalid_base64_content',
            'isBase64Encoded': True,
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {'TWILIO_AUTH_TOKEN': 'test_token'}, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                response = webhook_handler(event, context)
        
        assert response['statusCode'] == 400
        assert 'Bad Request: Invalid body encoding' in response['body']
    
    def test_lambda_invocation_failure(self):
        """Test handling of Lambda invocation failure."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': 'From=%2B15551234567&To=%2B15557654321&Body=test',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_lambda_client.invoke.side_effect = Exception('Lambda invocation failed')
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        assert response['statusCode'] == 500
        assert 'Could not start identification process' in response['body']
    
    def test_url_reconstruction_with_query_string(self):
        """Test URL reconstruction with query string."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': 'From=%2B15551234567&To=%2B15557654321&Body=test',
            'rawPath': '/webhook',
            'rawQueryString': 'param1=value1&param2=value2'
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        # Verify the URL was reconstructed correctly
        expected_url = 'https://test.com/webhook?param1=value1&param2=value2'
        mock_validator.return_value.validate.assert_called_once()
        call_args = mock_validator.return_value.validate.call_args[0]
        assert call_args[0] == expected_url
    
    def test_header_case_insensitive(self):
        """Test that headers are handled case-insensitively."""
        event = {
            'headers': {
                'Host': 'test.com',
                'X-Twilio-Signature': 'valid_signature',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            'body': 'From=%2B15551234567&To=%2B15557654321&Body=test',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        assert response['statusCode'] == 200
        mock_lambda_client.invoke.assert_called_once()
    
    def test_empty_body_handling(self):
        """Test handling of empty body."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': '',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        assert response['statusCode'] == 200
        # Verify empty body was handled correctly
        mock_validator.return_value.validate.assert_called_once()
        call_args = mock_validator.return_value.validate.call_args[0]
        assert call_args[1] == {}  # Empty post vars
    
    def test_consent_message_included(self):
        """Test that consent message is included in response."""
        event = {
            'headers': {'host': 'test.com', 'x-twilio-signature': 'valid_signature'},
            'body': 'From=%2B15551234567&To=%2B15557654321&Body=test',
            'rawPath': '/webhook',
            'rawQueryString': ''
        }
        context = {}
        
        with patch.dict(os.environ, {
            'TWILIO_AUTH_TOKEN': 'test_token',
            'PROCESSOR_LAMBDA_NAME': 'test-processor'
        }, clear=True):
            with patch('webhook_responder.RequestValidator') as mock_validator:
                mock_validator.return_value.validate.return_value = True
                
                with patch('webhook_responder.boto3.client') as mock_boto3:
                    mock_lambda_client = Mock()
                    mock_boto3.return_value = mock_lambda_client
                    
                    response = webhook_handler(event, context)
        
        assert response['statusCode'] == 200
        body = response['body']
        assert 'consent to receive a response' in body
        assert 'Respond STOP to stop' in body
        assert 'Little Tree Farms Horse ID' in body