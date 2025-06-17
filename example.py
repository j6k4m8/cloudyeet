#!/usr/bin/env python3
"""
Example usage of cloudyeet with service permissions.
"""
import boto3
from cloudyeet import lambda_yeet

# Example 1: Function that needs S3 access
@lambda_yeet(
    function_name="s3-processor",
    services=["s3"],
    timeout_sec=30
)
def process_s3_file(bucket: str, key: str):
    """Process a file from S3."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj['Body'].read().decode('utf-8')
    return f"Processed file {key} with {len(content)} characters"

# Example 2: Function that needs SQS and DynamoDB access
@lambda_yeet(
    function_name="queue-processor",
    services=["sqs", "dynamodb"],
    memory_mb=256
)
def process_queue_message(queue_url: str, table_name: str):
    """Process messages from SQS and store results in DynamoDB."""
    sqs = boto3.client('sqs')
    dynamodb = boto3.resource('dynamodb')

    # Receive message from SQS
    response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
    messages = response.get('Messages', [])

    if messages:
        message = messages[0]
        # Store in DynamoDB
        table = dynamodb.Table(table_name)
        table.put_item(Item={
            'id': message['MessageId'],
            'body': message['Body'],
            'processed_at': str(message['ReceiptHandle'])
        })

        # Delete from SQS
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=message['ReceiptHandle']
        )
        return f"Processed message {message['MessageId']}"

    return "No messages to process"

# Example 3: Function with multiple AWS services
@lambda_yeet(
    function_name="multi-service-processor",
    services=["s3", "sns", "secretsmanager"],
    requirements=["boto3"]
)
def notify_on_file_upload(bucket: str, key: str, topic_arn: str):
    """Process file upload and send notification."""
    import boto3

    s3 = boto3.client('s3')
    sns = boto3.client('sns')
    secrets = boto3.client('secretsmanager')

    # Get file metadata
    obj_info = s3.head_object(Bucket=bucket, Key=key)
    file_size = obj_info['ContentLength']

    # Send notification
    message = f"New file uploaded: {key} ({file_size} bytes)"
    sns.publish(TopicArn=topic_arn, Message=message)

    return f"Notification sent for {key}"

if __name__ == "__main__":
    # Example usage (these would require actual AWS resources)
    print("CloudYeet with service permissions examples:")
    print("1. S3 processor function created")
    print("2. SQS + DynamoDB processor function created")
    print("3. Multi-service processor function created")

    # Uncomment to test (requires AWS credentials and resources):
    # result = process_s3_file("my-bucket", "test-file.txt")
    # print(f"Result: {result}")
