import base64
import boto3
import cloudpickle
import functools
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from typing import Callable, Optional

# Get region from environment or fall back to us-east-1
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

lambda_client = boto3.client("lambda", region_name=AWS_REGION)
iam_client = boto3.client("iam", region_name=AWS_REGION)

# Track uploaded versions
lambda_registry = {}


def _hash_blob(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _create_service_policy(services: list) -> dict:
    """Create IAM policy for specified AWS services."""
    service_actions = {
        "s3": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
        "sqs": [
            "sqs:SendMessage",
            "sqs:ReceiveMessage",
            "sqs:DeleteMessage",
            "sqs:GetQueueAttributes",
        ],
        "dynamodb": [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:DeleteItem",
            "dynamodb:Query",
            "dynamodb:Scan",
        ],
        "sns": ["sns:Publish", "sns:Subscribe", "sns:Unsubscribe"],
        "ses": ["ses:SendEmail", "ses:SendRawEmail"],
        "secretsmanager": ["secretsmanager:GetSecretValue"],
        "ssm": ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
        "cloudwatch": [
            "cloudwatch:PutMetricData",
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents",
        ],
    }

    actions = []
    for service in services:
        if service in service_actions:
            actions.extend(service_actions[service])

    return {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": actions, "Resource": "*"}],
    }


def _get_role_arn(
    role_name="CloudYeetExecutionRole", services: Optional[list] = None
) -> str:
    assume_role_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )
    try:
        role = iam_client.create_role(
            RoleName=role_name, AssumeRolePolicyDocument=assume_role_policy
        )
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        # Attach service-specific policies if services are specified
        if services:
            policy_name = f"{role_name}ServicePolicy"
            service_policy = _create_service_policy(services)
            try:
                iam_client.put_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(service_policy),
                )
            except Exception:
                # Policy might already exist, update it
                iam_client.put_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(service_policy),
                )

        # Wait for role to propagate
        time.sleep(5)
        return role["Role"]["Arn"]
    except iam_client.exceptions.EntityAlreadyExistsException:
        role_arn = iam_client.get_role(RoleName=role_name)["Role"]["Arn"]

        # Update service policies for existing role if services are specified
        if services:
            policy_name = f"{role_name}ServicePolicy"
            service_policy = _create_service_policy(services)
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(service_policy),
            )

        return role_arn


def _create_lambda_zip(
    serialized_func: bytes, requirements: Optional[list] = None
) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        handler_code = """
import cloudpickle
import base64

def lambda_handler(event, context):
    payload = event.get('payload')
    args = event.get('args', [])
    kwargs = event.get('kwargs', {})
    func = cloudpickle.loads(base64.b64decode(payload))
    return func(*args, **kwargs)
"""
        lambda_file = os.path.join(tmpdir, "lambda_function.py")
        with open(lambda_file, "w") as f:
            f.write(handler_code)

        if requirements:
            package_dir = os.path.join(tmpdir, "package")
            os.makedirs(package_dir, exist_ok=True)

            # Copy lambda function to package directory
            import shutil

            shutil.copy2(lambda_file, package_dir)

            # Create requirements file
            req_file = os.path.join(tmpdir, "requirements.txt")

            # Filter out large packages that should be avoided or handled differently
            heavy_packages = {
                "numpy",
                "scipy",
                "matplotlib",
                "pandas",
                "tensorflow",
                "torch",
                "opencv-python",
                "orjson",  # Has native binary components, needs Linux targeting
            }
            light_requirements = []
            heavy_requirements = []

            for req in requirements:
                package_name = (
                    req.split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split("~=")[0]
                    .strip()
                )
                if package_name.lower() in heavy_packages:
                    heavy_requirements.append(req)
                else:
                    light_requirements.append(req)

            if heavy_requirements:
                print(f"Warning: Heavy packages detected: {heavy_requirements}")
                print(
                    "These may cause Lambda size limits. Consider using Lambda layers."
                )

            # Install packages using available pip
            # Try multiple pip commands in order of preference
            pip_commands_to_try = [
                ["python", "-m", "pip"],  # Most reliable
                ["python3", "-m", "pip"],  # Common alternative
                ["uv", "pip"],  # UV's pip interface
                ["pip"],  # Direct pip if available
            ]

            pip_cmd = None
            for cmd_base in pip_commands_to_try:
                try:
                    # Test if command exists
                    test_result = subprocess.run(
                        cmd_base + ["--version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if test_result.returncode == 0:
                        pip_cmd = cmd_base
                        break
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    continue

            if not pip_cmd:
                raise RuntimeError(
                    "No pip command found. Please ensure pip is installed."
                )  # Install light packages first
        if light_requirements:
            with open(req_file, "w") as f:
                f.write("\n".join(light_requirements))

            # Build install command with manylinux for ALL packages
            install_cmd = pip_cmd + [
                "install",
                "-r",
                req_file,
                "--target",
                package_dir,
                "--no-cache-dir",
                "--platform",
                "manylinux2014_x86_64",
                "--implementation",
                "cp",
                "--python-version",
                "3.11",
                "--only-binary=:all:",
                "--upgrade",
            ]

            try:
                subprocess.run(install_cmd, capture_output=True, text=True, check=True)
                print(
                    f"Successfully installed {len(light_requirements)} light packages with manylinux2014"
                )
            except subprocess.CalledProcessError as e:
                print(f"Light packages manylinux install failed: {e.stderr}")
                # Fallback: try with linux_x86_64
                fallback_cmd = pip_cmd + [
                    "install",
                    "-r",
                    req_file,
                    "--target",
                    package_dir,
                    "--no-cache-dir",
                    "--platform",
                    "linux_x86_64",
                    "--implementation",
                    "cp",
                    "--python-version",
                    "3.11",
                    "--only-binary=:all:",
                    "--upgrade",
                ]
                try:
                    subprocess.run(
                        fallback_cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        f"Successfully installed {len(light_requirements)} light packages with linux_x86_64"
                    )
                except subprocess.CalledProcessError as e2:
                    print(f"Light packages linux_x86_64 install failed: {e2.stderr}")
                    # Final fallback: no platform restrictions
                    basic_cmd = pip_cmd + [
                        "install",
                        "-r",
                        req_file,
                        "--target",
                        package_dir,
                        "--no-cache-dir",
                        "--python-version",
                        "3.11",
                    ]
                    subprocess.run(
                        basic_cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        f"Successfully installed {len(light_requirements)} light packages without platform targeting"
                    )

            # Install heavy packages with manylinux strategy (same as slim handler)
            if heavy_requirements:
                print(
                    f"Installing {len(heavy_requirements)} heavy packages for fat Lambda..."
                )
                heavy_req_file = os.path.join(tmpdir, "heavy_requirements.txt")
                with open(heavy_req_file, "w") as f:
                    f.write("\n".join(heavy_requirements))

                success = False

                # Strategy 1: Use manylinux2014 (recommended by Pydantic docs)
                try:
                    strategy1_cmd = [
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        heavy_req_file,
                        "--target",
                        package_dir,
                        "--no-cache-dir",
                        "--platform",
                        "manylinux2014_x86_64",
                        "--implementation",
                        "cp",
                        "--python-version",
                        "3.11",
                        "--only-binary=:all:",
                        "--upgrade",
                    ]
                    subprocess.run(
                        strategy1_cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        "Successfully installed heavy packages with manylinux2014 wheels"
                    )
                    success = True
                except subprocess.CalledProcessError as e:
                    print(f"Strategy 1 (manylinux2014) failed: {e.stderr}")

                # Strategy 2: Try with linux_x86_64 platform (fallback)
                if not success:
                    try:
                        strategy2_cmd = [
                            "python",
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            heavy_req_file,
                            "--target",
                            package_dir,
                            "--no-cache-dir",
                            "--platform",
                            "linux_x86_64",
                            "--implementation",
                            "cp",
                            "--python-version",
                            "3.11",
                            "--only-binary=:all:",
                            "--upgrade",
                        ]
                        subprocess.run(
                            strategy2_cmd, capture_output=True, text=True, check=True
                        )
                        print(
                            "Successfully installed heavy packages with linux_x86_64 platform"
                        )
                        success = True
                    except subprocess.CalledProcessError as e:
                        print(f"Strategy 2 (linux_x86_64) failed: {e.stderr}")

                # Strategy 3: Try each package individually
                if not success:
                    print("Trying to install heavy packages individually...")
                    successful_packages = []
                    for req in heavy_requirements:
                        package_installed = False

                        # Try with manylinux2014 first
                        try:
                            individual_cmd = [
                                "python",
                                "-m",
                                "pip",
                                "install",
                                req,
                                "--target",
                                package_dir,
                                "--no-cache-dir",
                                "--platform",
                                "manylinux2014_x86_64",
                                "--implementation",
                                "cp",
                                "--python-version",
                                "3.11",
                                "--only-binary=:all:",
                                "--upgrade",
                            ]
                            subprocess.run(
                                individual_cmd,
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            successful_packages.append(req)
                            package_installed = True
                            print(f"✓ Installed {req} (manylinux2014)")
                        except subprocess.CalledProcessError:
                            pass

                        # Fallback to linux_x86_64
                        if not package_installed:
                            try:
                                linux_cmd = [
                                    "python",
                                    "-m",
                                    "pip",
                                    "install",
                                    req,
                                    "--target",
                                    package_dir,
                                    "--no-cache-dir",
                                    "--platform",
                                    "linux_x86_64",
                                    "--implementation",
                                    "cp",
                                    "--python-version",
                                    "3.11",
                                    "--only-binary=:all:",
                                    "--upgrade",
                                ]
                                subprocess.run(
                                    linux_cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                )
                                successful_packages.append(req)
                                package_installed = True
                                print(f"✓ Installed {req} (linux_x86_64)")
                            except subprocess.CalledProcessError:
                                pass

                        if not package_installed:
                            print(f"✗ Failed to install {req}")

                    if successful_packages:
                        success = True
                        print(
                            f"Successfully installed {len(successful_packages)} out of {len(heavy_requirements)} heavy packages"
                        )

                if not success:
                    print(
                        "Warning: Could not install any heavy packages with Linux compatibility"
                    )
                    print("Creating warning file instead...")
                    # Create warning file as fallback
                    warning_file = os.path.join(
                        package_dir, "HEAVY_PACKAGES_EXCLUDED.txt"
                    )
                    with open(warning_file, "w") as f:
                        f.write(
                            "The following heavy packages could not be installed:\n"
                        )
                        f.write("\n".join(heavy_requirements))
                        f.write("\n\nLambda function may fail without these packages.")

            # Remove unnecessary directories to save space
            for root, dirs, files in os.walk(package_dir):
                # Remove __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                # Remove .pyc files
                for file in files:
                    if file.endswith(".pyc"):
                        os.remove(os.path.join(root, file))

            # Create zip from package directory
            zip_path = os.path.join(tmpdir, "function.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, package_dir)
                        zipf.write(file_path, arc_name)
        else:
            # Simple case: just zip the lambda function
            zip_path = os.path.join(tmpdir, "function.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(lambda_file, "lambda_function.py")

        with open(zip_path, "rb") as f:
            return f.read()


def _update_existing_function(
    function_name: str, zip_bytes: bytes, timeout_sec: int, memory_mb: int
) -> None:
    """Update existing Lambda function with retry logic for concurrent operations."""
    max_retries = 5
    base_delay = 2

    # Update function code first
    for attempt in range(max_retries):
        try:
            lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_bytes,
            )
            break
        except lambda_client.exceptions.ResourceConflictException as e:
            if "update is in progress" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                time.sleep(delay)
                continue
            raise

    # Wait a moment before updating configuration
    time.sleep(1)

    # Update function configuration
    for attempt in range(max_retries):
        try:
            lambda_client.update_function_configuration(
                FunctionName=function_name,
                Timeout=timeout_sec,
                MemorySize=memory_mb,
            )
            break
        except lambda_client.exceptions.ResourceConflictException as e:
            if "update is in progress" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                time.sleep(delay)
                continue
            raise


def _create_slim_s3_handler(
    function_name: str,
    func: Callable,
    requirements: Optional[list],
    services: Optional[list],
    timeout_sec: int,
    memory_mb: int,
):
    """Create a slim Lambda that downloads dependencies from S3 at runtime."""

    # Create S3 bucket for dependencies if it doesn't exist
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket_name = f"cloudyeet-deps-{AWS_REGION}"

    try:
        s3.create_bucket(Bucket=bucket_name)
    except s3.exceptions.BucketAlreadyExists:
        pass  # Bucket exists, that's fine
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass  # We own it, that's fine

    # Create dependencies bundle and upload to S3
    deps_key = None
    if requirements:
        deps_key = f"{function_name}/dependencies.zip"
        deps_zip = _create_dependencies_bundle(requirements)
        s3.put_object(Bucket=bucket_name, Key=deps_key, Body=deps_zip)
        print(f"Uploaded dependencies bundle to s3://{bucket_name}/{deps_key}")

    # Create slim Lambda handler that downloads deps at runtime
    payload = base64.b64encode(cloudpickle.dumps(func)).decode()

    # Create the handler code without embedding the huge payload as a literal string
    slim_handler_code = f'''
import json
import boto3
import zipfile
import sys
import os
import tempfile
import base64
import cloudpickle

# Dependencies configuration
DEPS_BUCKET = "{bucket_name}"
DEPS_KEY = {repr(deps_key)}

def lambda_handler(event, context):
    # Download and extract dependencies from S3 FIRST if needed
    if DEPS_KEY:
        deps_dir = "/tmp/deps"
        if not os.path.exists(deps_dir):
            s3 = boto3.client("s3")

            # Download dependencies bundle
            with tempfile.NamedTemporaryFile() as tmp_file:
                s3.download_fileobj(DEPS_BUCKET, DEPS_KEY, tmp_file)
                tmp_file.seek(0)

                # Extract to /tmp/deps
                with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
                    zip_ref.extractall(deps_dir)

            # Add to Python path BEFORE unpickling
            sys.path.insert(0, deps_dir)

    # NOW we can safely unpickle and execute the function
    payload = event.get('payload')
    args = event.get('args', [])
    kwargs = event.get('kwargs', {{}})

    # Unpickle AFTER dependencies are available
    func = cloudpickle.loads(base64.b64decode(payload))
    return func(*args, **kwargs)
'''

    # Create minimal Lambda zip with just the handler
    zip_bytes = _create_minimal_lambda_zip(slim_handler_code)

    # Deploy the slim Lambda
    role_arn = _get_role_arn(services=(services or []) + ["s3"])

    try:
        lambda_client.create_function(
            FunctionName=function_name,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=timeout_sec,
            MemorySize=memory_mb,
        )
    except lambda_client.exceptions.ResourceConflictException:
        # Function exists, update it with retry logic
        _update_existing_function(function_name, zip_bytes, timeout_sec, memory_mb)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {
                    "payload": payload,  # Pass the pickled function in the event
                    "args": args,
                    "kwargs": kwargs,
                }
            ),
        )
        return json.loads(response["Payload"].read())

    return wrapper


def _create_dependencies_bundle(requirements: list) -> bytes:
    """Create a zip bundle of all dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        deps_dir = os.path.join(tmpdir, "deps")
        os.makedirs(deps_dir, exist_ok=True)

        # Create requirements file - separate heavy and light packages for better handling
        req_file = os.path.join(tmpdir, "requirements.txt")
        heavy_req_file = os.path.join(tmpdir, "heavy_requirements.txt")

        # Heavy packages that often have platform-specific binaries
        heavy_packages = {
            "numpy",
            "scipy",
            "matplotlib",
            "pandas",
            "tensorflow",
            "torch",
            "opencv-python",
            "scikit-learn",
            "lxml",
            "psutil",
            "orjson",  # Has native binary components, needs Linux targeting
        }

        light_requirements = []
        heavy_requirements = []

        for req in requirements:
            package_name = (
                req.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
            )
            if package_name.lower() in heavy_packages:
                heavy_requirements.append(req)
            else:
                light_requirements.append(req)

        # Write light requirements (no platform restrictions needed)
        with open(req_file, "w") as f:
            f.write("\n".join(light_requirements))

        # Write heavy requirements separately
        with open(heavy_req_file, "w") as f:
            f.write("\n".join(heavy_requirements))

        # Install packages using available pip
        pip_commands_to_try = [
            ["python", "-m", "pip"],
            ["python3", "-m", "pip"],
            ["uv", "pip"],
            ["pip"],
        ]

        pip_cmd = None
        for cmd_base in pip_commands_to_try:
            try:
                test_result = subprocess.run(
                    cmd_base + ["--version"], capture_output=True, text=True, timeout=10
                )
                if test_result.returncode == 0:
                    pip_cmd = cmd_base
                    break
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                continue

        if not pip_cmd:
            raise RuntimeError("No pip command found. Please ensure pip is installed.")

        # Install light packages first with manylinux targeting
        if light_requirements:
            light_install_cmd = pip_cmd + [
                "install",
                "-r",
                req_file,
                "--target",
                deps_dir,
                "--no-cache-dir",
                "--platform",
                "manylinux2014_x86_64",
                "--implementation",
                "cp",
                "--python-version",
                "3.11",
                "--only-binary=:all:",
                "--upgrade",
            ]
            try:
                subprocess.run(
                    light_install_cmd, capture_output=True, text=True, check=True
                )
                print(
                    f"Successfully installed {len(light_requirements)} light packages with manylinux2014"
                )
            except subprocess.CalledProcessError as e:
                print(f"Light packages manylinux install failed: {e.stderr}")
                # Fallback: try with linux_x86_64
                fallback_cmd = pip_cmd + [
                    "install",
                    "-r",
                    req_file,
                    "--target",
                    deps_dir,
                    "--no-cache-dir",
                    "--platform",
                    "linux_x86_64",
                    "--implementation",
                    "cp",
                    "--python-version",
                    "3.11",
                    "--only-binary=:all:",
                    "--upgrade",
                ]
                try:
                    subprocess.run(
                        fallback_cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        f"Successfully installed {len(light_requirements)} light packages with linux_x86_64"
                    )
                except subprocess.CalledProcessError as e2:
                    print(f"Light packages linux_x86_64 install failed: {e2.stderr}")
                    # Final fallback: no platform restrictions
                    basic_cmd = pip_cmd + [
                        "install",
                        "-r",
                        req_file,
                        "--target",
                        deps_dir,
                        "--no-cache-dir",
                        "--python-version",
                        "3.11",
                    ]
                    subprocess.run(
                        basic_cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        f"Successfully installed {len(light_requirements)} light packages without platform targeting"
                    )  # Install heavy packages with Linux platform targeting
        if heavy_requirements:
            print(f"Installing {len(heavy_requirements)} heavy packages for Lambda...")
            success = False

            # Strategy 1: Use manylinux2014 (recommended by Pydantic docs)
            try:
                strategy1_cmd = [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    heavy_req_file,
                    "--target",
                    deps_dir,
                    "--no-cache-dir",
                    "--platform",
                    "manylinux2014_x86_64",
                    "--implementation",
                    "cp",
                    "--python-version",
                    "3.11",
                    "--only-binary=:all:",
                    "--upgrade",
                ]
                subprocess.run(
                    strategy1_cmd, capture_output=True, text=True, check=True
                )
                print("Successfully installed heavy packages with manylinux2014 wheels")
                success = True
            except subprocess.CalledProcessError as e:
                print(f"Strategy 1 (manylinux2014) failed: {e.stderr}")

            # Strategy 2: Try with linux_x86_64 platform (fallback)
            if not success:
                try:
                    strategy2_cmd = [
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        heavy_req_file,
                        "--target",
                        deps_dir,
                        "--no-cache-dir",
                        "--platform",
                        "linux_x86_64",
                        "--implementation",
                        "cp",
                        "--python-version",
                        "3.11",
                        "--only-binary=:all:",
                        "--upgrade",
                    ]
                    subprocess.run(
                        strategy2_cmd, capture_output=True, text=True, check=True
                    )
                    print(
                        "Successfully installed heavy packages with linux_x86_64 platform"
                    )
                    success = True
                except subprocess.CalledProcessError as e:
                    print(f"Strategy 2 (linux_x86_64) failed: {e.stderr}")

            # Strategy 3: Try each package individually with manylinux2014 first
            if not success:
                print("Trying to install heavy packages individually...")
                successful_packages = []
                for req in heavy_requirements:
                    package_installed = False

                    # Try with manylinux2014 first (Pydantic recommended approach)
                    try:
                        individual_cmd = [
                            "python",
                            "-m",
                            "pip",
                            "install",
                            req,
                            "--target",
                            deps_dir,
                            "--no-cache-dir",
                            "--platform",
                            "manylinux2014_x86_64",
                            "--implementation",
                            "cp",
                            "--python-version",
                            "3.11",
                            "--only-binary=:all:",
                            "--upgrade",
                        ]
                        subprocess.run(
                            individual_cmd, capture_output=True, text=True, check=True
                        )
                        successful_packages.append(req)
                        package_installed = True
                        print(f"✓ Installed {req} (manylinux2014)")
                    except subprocess.CalledProcessError:
                        pass

                    # Fallback to linux_x86_64 if manylinux2014 failed
                    if not package_installed:
                        try:
                            linux_cmd = [
                                "python",
                                "-m",
                                "pip",
                                "install",
                                req,
                                "--target",
                                deps_dir,
                                "--no-cache-dir",
                                "--platform",
                                "linux_x86_64",
                                "--implementation",
                                "cp",
                                "--python-version",
                                "3.11",
                                "--only-binary=:all:",
                                "--upgrade",
                            ]
                            subprocess.run(
                                linux_cmd, capture_output=True, text=True, check=True
                            )
                            successful_packages.append(req)
                            package_installed = True
                            print(f"✓ Installed {req} (linux_x86_64)")
                        except subprocess.CalledProcessError:
                            pass

                    if not package_installed:
                        print(f"✗ Failed to install {req}")

                if successful_packages:
                    success = True
                    print(
                        f"Successfully installed {len(successful_packages)} out of {len(heavy_requirements)} heavy packages"
                    )

            if not success:
                print(
                    "Warning: Could not install any heavy packages with Linux compatibility"
                )
                print("Lambda function may fail if these packages are required")

        # Clean up problematic directories that can cause import issues
        for root, dirs, files in os.walk(deps_dir):
            # Remove source directories that can cause numpy import errors
            for dir_name in list(dirs):
                if dir_name in ["tests", "test", "__pycache__", ".git", "doc", "docs"]:
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path, ignore_errors=True)

            # Remove problematic files
            for file_name in list(files):
                if (
                    file_name.endswith(
                        (".c", ".cpp", ".h", ".f", ".f90", ".pyx", ".pxd")
                    )
                    or file_name in ["setup.py", "setup.cfg", "pyproject.toml"]
                    or file_name.startswith("test_")
                ):
                    file_path = os.path.join(root, file_name)
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

        # Create zip of dependencies
        zip_path = os.path.join(tmpdir, "deps.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(deps_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, deps_dir)
                    zipf.write(file_path, arc_name)

        with open(zip_path, "rb") as f:
            return f.read()


def _create_minimal_lambda_zip(handler_code: str) -> bytes:
    """Create a minimal zip with just the Lambda handler and cloudpickle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lambda_file = os.path.join(tmpdir, "lambda_function.py")
        with open(lambda_file, "w") as f:
            f.write(handler_code)

        # Install just cloudpickle to the zip with Python version
        try:
            subprocess.run(
                [
                    "pip",
                    "install",
                    "cloudpickle",
                    "--target",
                    tmpdir,
                    "--no-cache-dir",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to python -m pip
            try:
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "cloudpickle",
                        "--target",
                        tmpdir,
                        "--no-cache-dir",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not install cloudpickle: {e.stderr}")
                print("Function may fail without cloudpickle")

        zip_path = os.path.join(tmpdir, "function.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(tmpdir):
                # Skip the zip file itself
                if root == tmpdir and "function.zip" in files:
                    files.remove("function.zip")
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, tmpdir)
                    zipf.write(file_path, arc_name)

        with open(zip_path, "rb") as f:
            return f.read()


def lambda_yeet(
    function_name: str,
    timeout_sec: int = 60,
    memory_mb: int = 128,
    requirements: Optional[list] = None,
    services: Optional[list] = None,
    slim_handler: bool = False,
):
    def decorator(func: Callable):
        if slim_handler:
            # Slim handler: upload dependencies to S3, download at runtime
            return _create_slim_s3_handler(
                function_name, func, requirements, services, timeout_sec, memory_mb
            )

        # Regular approach: pack everything into Lambda
        payload = base64.b64encode(cloudpickle.dumps(func)).decode()
        blob_hash = _hash_blob(payload.encode())

        if lambda_registry.get(function_name) != blob_hash:
            role_arn = _get_role_arn(services=services)
            zip_bytes = _create_lambda_zip(payload.encode(), requirements)

            # Retry lambda creation with backoff for role propagation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime="python3.11",
                        Role=role_arn,
                        Handler="lambda_function.lambda_handler",
                        Code={"ZipFile": zip_bytes},
                        Timeout=timeout_sec,
                        MemorySize=memory_mb,
                    )
                    break
                except lambda_client.exceptions.InvalidParameterValueException as e:
                    if (
                        "cannot be assumed by Lambda" in str(e)
                        and attempt < max_retries - 1
                    ):
                        time.sleep(5)  # Wait for role propagation
                        continue
                    raise
                except lambda_client.exceptions.ResourceConflictException:
                    # Function exists, update it with retry logic
                    _update_existing_function(
                        function_name, zip_bytes, timeout_sec, memory_mb
                    )
                    break
            lambda_registry[function_name] = blob_hash

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(
                    {"payload": payload, "args": args, "kwargs": kwargs}
                ),
            )
            return json.loads(response["Payload"].read())

        return wrapper

    return decorator


def lambda_delete(name: str):
    if name in lambda_registry:
        lambda_client.delete_function(FunctionName=name)
        del lambda_registry[name]
