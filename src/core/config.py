"""
Configuration management for the agentic framework testing harness.
"""

import os
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """
    Central configuration manager for the testing harness.
    
    Loads environment variables and provides framework-specific configurations.
    """
    
    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_path: Optional path to .env file
        """
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        # API Keys
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'azure_openai': os.getenv('AZURE_OPENAI_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY'),
            'cohere': os.getenv('COHERE_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY')
        }
        
        # Model Configuration
        self.default_model = os.getenv('DEFAULT_MODEL', 'gpt-4')
        self.default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        self.default_max_tokens = int(os.getenv('DEFAULT_MAX_TOKENS', '2048'))
        
        # Framework-specific models
        self.framework_models = {
            'langgraph': os.getenv('LANGGRAPH_MODEL', self.default_model),
            'crewai': os.getenv('CREWAI_MODEL', self.default_model),
            'autogen': os.getenv('AUTOGEN_MODEL', self.default_model)
        }
        
        # Testing Configuration
        self.test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'
        self.max_test_cases = int(os.getenv('MAX_TEST_CASES', '10'))
        self.timeout_seconds = int(os.getenv('TIMEOUT_SECONDS', '300'))
        
        # Reporting
        self.report_output_dir = Path(os.getenv('REPORT_OUTPUT_DIR', 'benchmark_results'))
        self.generate_charts = os.getenv('GENERATE_CHARTS', 'true').lower() == 'true'
        self.verbose = os.getenv('VERBOSE', 'false').lower() == 'true'
        
        # AWS Configuration (for Bedrock and Ground Truth)
        self.aws_bedrock_enabled = os.getenv('AWS_BEDROCK_ENABLED', 'false').lower() == 'true'
        self.aws_ground_truth_enabled = os.getenv('AWS_GROUND_TRUTH_ENABLED', 'false').lower() == 'true'
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.aws_session_token = os.getenv('AWS_SESSION_TOKEN')  # Optional for temporary credentials
        
        # If AWS credentials are present, enable Bedrock automatically
        if self.aws_access_key_id and self.aws_secret_access_key:
            self.aws_bedrock_enabled = True
        
        # Ground Truth specific settings
        self.ground_truth_config = {
            'enabled': self.aws_ground_truth_enabled,
            'role_arn': os.getenv('GROUND_TRUTH_ROLE_ARN'),  # IAM role for Ground Truth
            's3_input_bucket': os.getenv('GROUND_TRUTH_S3_INPUT_BUCKET'),
            's3_output_bucket': os.getenv('GROUND_TRUTH_S3_OUTPUT_BUCKET'),
            'labeling_job_name_prefix': os.getenv('GROUND_TRUTH_JOB_PREFIX', 'agentic-framework-eval'),
            'workteam_arn': os.getenv('GROUND_TRUTH_WORKTEAM_ARN'),  # Private workforce ARN
            'use_automated_labeling': os.getenv('GROUND_TRUTH_AUTO_LABELING', 'false').lower() == 'true',
            'label_category_config': os.getenv('GROUND_TRUTH_LABEL_CATEGORY', 'agent-evaluation'),
            'max_human_labeled_objects': int(os.getenv('GROUND_TRUTH_MAX_HUMAN_LABELS', '1000')),
            'max_percent_objects': int(os.getenv('GROUND_TRUTH_MAX_PERCENT_OBJECTS', '100')),
            'enable_encryption': os.getenv('GROUND_TRUTH_ENABLE_ENCRYPTION', 'true').lower() == 'true',
            'kms_key_id': os.getenv('GROUND_TRUTH_KMS_KEY_ID'),  # Optional KMS key for encryption
        }
        
        # Ground Truth evaluation types
        self.ground_truth_evaluation_types = {
            'accuracy_scoring': os.getenv('GT_EVAL_ACCURACY', 'true').lower() == 'true',
            'quality_assessment': os.getenv('GT_EVAL_QUALITY', 'true').lower() == 'true',
            'human_preference': os.getenv('GT_EVAL_PREFERENCE', 'false').lower() == 'true',
            'task_completion': os.getenv('GT_EVAL_COMPLETION', 'true').lower() == 'true',
            'safety_review': os.getenv('GT_EVAL_SAFETY', 'false').lower() == 'true'
        }
        
        # Ground Truth labeling job templates
        self.ground_truth_templates = {
            'text_classification': os.getenv('GT_TEMPLATE_CLASSIFICATION', 's3://aws-sagemaker-ground-truth-templates/text-classification'),
            'named_entity_recognition': os.getenv('GT_TEMPLATE_NER', 's3://aws-sagemaker-ground-truth-templates/ner'),
            'custom_evaluation': os.getenv('GT_TEMPLATE_CUSTOM'),  # Custom template S3 URI
            'comparison_ranking': os.getenv('GT_TEMPLATE_RANKING'),  # For comparing framework outputs
        }
        
        # Ground Truth integration settings
        self.ground_truth_integration = {
            'auto_submit_results': os.getenv('GT_AUTO_SUBMIT', 'false').lower() == 'true',
            'batch_size': int(os.getenv('GT_BATCH_SIZE', '100')),
            'wait_for_completion': os.getenv('GT_WAIT_COMPLETION', 'false').lower() == 'true',
            'max_wait_time_hours': int(os.getenv('GT_MAX_WAIT_HOURS', '24')),
            'notification_topic_arn': os.getenv('GT_SNS_TOPIC_ARN'),  # SNS topic for notifications
            'use_consolidated_annotations': os.getenv('GT_USE_CONSOLIDATED', 'true').lower() == 'true'
        }
        
        # Create output directory if it doesn't exist
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self, provider: str) -> str:
        """
        Get API key for a provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            API key string
            
        Raises:
            ValueError: If API key not found
        """
        key = self.api_keys.get(provider.lower())
        if not key:
            raise ValueError(f"API key for {provider} not found. Please set it in .env file.")
        return key
    
    def get_framework_config(self, framework: str) -> Dict[str, Any]:
        """
        Get configuration for a specific framework.
        
        Args:
            framework: Framework name
            
        Returns:
            Framework-specific configuration dict
        """
        framework = framework.lower()
        
        # Common configuration for all frameworks
        config = {
            'temperature': self.default_temperature,
            'max_tokens': self.default_max_tokens,
            'timeout_seconds': self.timeout_seconds,
            'test_mode': self.test_mode
        }
        
        # Framework-specific overrides
        if framework == 'langgraph':
            config.update({
                'model': self.framework_models.get('langgraph', self.default_model),
                'api_key': self.get_api_key('openai'),
                'streaming': False
            })
        elif framework == 'crewai':
            config.update({
                'model': self.framework_models.get('crewai', self.default_model),
                'api_key': self.get_api_key('openai'),
                'verbose': self.verbose,
                'max_iter': 10
            })
        elif framework == 'autogen':
            config.update({
                'model': self.framework_models.get('autogen', self.default_model),
                'api_key': self.get_api_key('openai'),
                'seed': 42,
                'max_consecutive_auto_reply': 10
            })
        elif framework == 'pydantic_ai':
            config.update({
                'model': self.default_model,
                'api_key': self.get_api_key('openai')
            })
        elif framework == 'haystack':
            config.update({
                'model': self.default_model,
                'api_key': self.get_api_key('openai')
            })
        elif framework == 'llamaindex':
            config.update({
                'model': self.default_model,
                'api_key': self.get_api_key('openai')
            })
        elif framework == 'dspy':
            config.update({
                'model': self.default_model,
                'api_key': self.get_api_key('openai')
            })
        else:
            # Default configuration for other frameworks
            config.update({
                'model': self.default_model,
                'api_key': self.get_api_key('openai')
            })
        
        return config
    
    def get_ground_truth_config(self) -> Dict[str, Any]:
        """
        Get AWS Ground Truth configuration.
        
        Returns:
            Ground Truth configuration dict
        """
        if not self.aws_ground_truth_enabled:
            return {'enabled': False}
        
        return {
            **self.ground_truth_config,
            'aws_config': {
                'region': self.aws_region,
                'access_key_id': self.aws_access_key_id,
                'secret_access_key': self.aws_secret_access_key,
                'session_token': self.aws_session_token
            },
            'evaluation_types': self.ground_truth_evaluation_types,
            'templates': self.ground_truth_templates,
            'integration': self.ground_truth_integration
        }
    
    def is_ground_truth_enabled(self) -> bool:
        """Check if AWS Ground Truth is enabled."""
        return self.aws_ground_truth_enabled
    
    def get_ground_truth_s3_paths(self) -> Dict[str, str]:
        """
        Get S3 paths for Ground Truth data.
        
        Returns:
            Dictionary with input and output S3 bucket paths
        """
        return {
            'input': f"s3://{self.ground_truth_config['s3_input_bucket']}",
            'output': f"s3://{self.ground_truth_config['s3_output_bucket']}"
        }
    
    def validate_ground_truth_config(self) -> bool:
        """
        Validate Ground Truth configuration.
        
        Returns:
            True if Ground Truth config is valid or disabled
            
        Raises:
            ValueError: If Ground Truth is enabled but config is invalid
        """
        if not self.aws_ground_truth_enabled:
            return True
        
        # Check AWS credentials
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials required when Ground Truth is enabled")
        
        # Check required Ground Truth settings
        if not self.ground_truth_config['role_arn']:
            raise ValueError("GROUND_TRUTH_ROLE_ARN required when Ground Truth is enabled")
        
        if not self.ground_truth_config['s3_input_bucket'] or not self.ground_truth_config['s3_output_bucket']:
            raise ValueError("S3 input and output buckets required for Ground Truth")
        
        if not self.ground_truth_config['workteam_arn'] and not self.ground_truth_config['use_automated_labeling']:
            raise ValueError("Either workteam ARN or automated labeling must be configured")
        
        return True
    
    def validate(self) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            True if all required configuration is valid
            
        Raises:
            ValueError: If required configuration is missing
        """
        # Check for at least one API key
        if not any(self.api_keys.values()):
            raise ValueError("At least one API key must be configured in .env file")
        
        # Check model configuration
        if not self.default_model:
            raise ValueError("DEFAULT_MODEL must be configured in .env file")
        
        # Check paths exist
        if not self.report_output_dir.exists():
            try:
                self.report_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create report output directory: {e}")
        
        # Validate Ground Truth config if enabled
        self.validate_ground_truth_config()
        
        return True
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  default_model={self.default_model},\n"
            f"  test_mode={self.test_mode},\n"
            f"  max_test_cases={self.max_test_cases},\n"
            f"  timeout_seconds={self.timeout_seconds},\n"
            f"  report_output_dir={self.report_output_dir},\n"
            f"  aws_ground_truth_enabled={self.aws_ground_truth_enabled},\n"
            f"  aws_region={self.aws_region if self.aws_ground_truth_enabled else 'N/A'}\n"
            f")"
        )
