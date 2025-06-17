# ☁️ cloudyeet

Turn any Python function into an AWS Lambda with a single decorator. Automatically deploys, updates, and tears down remote Lambdas — all from your local Python code.

## ✨ Features

-   `@lambda_yeet(...)` decorator auto-deploys functions to AWS Lambda
-   Detects code changes and **re-uploads** transparently
-   Serializes complex Python functions using `cloudpickle` (globals, closures, etc.)
-   Supports **custom memory, timeout, and pip dependencies**
-   `lambda_delete("function-name")` removes deployed Lambda
-   Uses `uv` for fast packaging and lightweight virtualenvs

---

## 🚀 Installation

```bash
uv venv
uv pip install -e .
```

Requires:

-   Python 3.8+
-   AWS credentials configured (via `~/.aws/credentials` or environment variables)
-   AWS Lambda + IAM permissions

---

## 🧪 Example Usage

```python
from cloudyeet import lambda_yeet, lambda_delete

@lambda_yeet(
    function_name="multiply-far-away",
    timeout_sec=90,
    memory_mb=256,
    requirements=["numpy"]
)
def remote_multiply(x, y):
    import numpy as np
    return int(np.array([x]) * y)[0]

# Call it — executes in AWS Lambda!
print(remote_multiply(6, 7))  # ➜ 42

# Tear down the Lambda when you're done
lambda_delete("multiply-far-away")
```

---

## 🛠️ API Reference

### `@lambda_yeet(...)`

Wrap your function with deployment metadata.

**Arguments**:

| Param           | Type        | Description                                  |
| --------------- | ----------- | -------------------------------------------- |
| `function_name` | `str`       | AWS Lambda function name                     |
| `timeout_sec`   | `int`       | Lambda timeout in seconds (default: `60`)    |
| `memory_mb`     | `int`       | Lambda memory in MB (default: `128`)         |
| `requirements`  | `list[str]` | Optional pip dependencies installed via `uv` |

---

### `lambda_delete(name: str)`

Tears down the corresponding Lambda function and clears local registry.

```python
lambda_delete("multiply-far-away")
```

---

## 🔐 AWS Permissions

You need permission to:

-   `lambda:CreateFunction`, `lambda:UpdateFunctionCode`, `lambda:DeleteFunction`
-   `iam:CreateRole`, `iam:AttachRolePolicy`, etc.

The package will create a role named `CloudYeetExecutionRole` if it doesn’t exist yet.

---

## 💡 How it Works

1. Your function is serialized using `cloudpickle`
2. A small `lambda_handler` is generated that unpickles and runs it
3. Packaged using `uv` and deployed via `boto3`
4. When the wrapped function is called, the Lambda is invoked with `args`/`kwargs`

---

## 🧹 Cleanup

Functions are tracked by hash. When you change your function, it's re-uploaded. To manually delete:

```python
lambda_delete("your-function-name")
```

---

## 📦 Project Structure

```
cloudyeet/
├── cloudyeet/
│   ├── __init__.py
│   └── core.py
├── pyproject.toml
└── README.md
```

## Examples

### Permissions

```python
# Function that needs S3 access
@lambda_yeet(
    function_name="s3-processor",
    services=["s3"],
    timeout_sec=30
)
def process_s3_file(bucket: str, key: str):
    # Your function can now access S3
    pass
```

```python
# Function that needs multiple services
@lambda_yeet(
    function_name="multi-service",
    services=["s3", "sqs", "dynamodb"],
    memory_mb=256
)
def complex_processor():
    # Function has access to S3, SQS, and DynamoDB
    pass
```

---

## 📜 License

Apache 2. Yeet responsibly.
