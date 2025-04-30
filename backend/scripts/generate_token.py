#!/usr/bin/env python
"""
Token Generator for Crocodile API

This script generates JWT tokens that can be used for authenticating with the Crocodile API.
It uses the JWT_SECRET_KEY from environment variables to ensure compatibility.

Usage:
  python generate_token.py [OPTIONS]

Options:
  --user-id TEXT    User ID to include in the token
  --email TEXT      Email to include in the token
  --name TEXT       Name to include in the token
  --role TEXT       Role to include in the token (default: user)
  --expires HOURS   Token expiration time in hours (default: 24)
  --json TEXT       Provide all token data as a JSON string
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from jose import jwt
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_token(data: Dict[str, Any], expires_delta: timedelta = timedelta(hours=24)) -> str:
    """
    Create a JWT token with the provided payload and expiration time.
    
    Args:
        data: Dictionary containing the token payload
        expires_delta: Token expiration time (default: 24 hours)
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    
    # Get JWT_SECRET_KEY from environment variables
    jwt_secret_key = os.environ.get("JWT_SECRET_KEY")
    
    if not jwt_secret_key:
        print("Error: JWT_SECRET_KEY environment variable is not set.")
        print("Please set JWT_SECRET_KEY in your environment or .env file.")
        sys.exit(1)
        
    encoded_jwt = jwt.encode(to_encode, jwt_secret_key, algorithm="HS256")
    return encoded_jwt

def main():
    parser = argparse.ArgumentParser(description="Generate JWT tokens for Crocodile API")
    parser.add_argument("--user-id", help="User ID to include in the token", default="user123")
    parser.add_argument("--email", help="Email to include in the token", default="user@example.com")
    parser.add_argument("--name", help="Name to include in the token", default="Test User")
    parser.add_argument("--role", help="Role to include in the token", default="user")
    parser.add_argument("--expires", type=int, help="Token expiration time in hours", default=24)
    parser.add_argument("--json", help="Provide all token data as a JSON string")
    
    args = parser.parse_args()
    
    # Default user data
    user_data = {
        "user_id": args.user_id,
        "email": args.email,
        "name": args.name,
        "role": args.role
    }
    
    # Override with JSON if provided
    if args.json:
        try:
            json_data = json.loads(args.json)
            user_data.update(json_data)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
            sys.exit(1)
    
    # Generate token
    token = create_token(user_data, timedelta(hours=args.expires))
    
    # Print the token and usage instructions
    print("\n=== JWT Token ===")
    print(token)
    print("\n=== Token Payload ===")
    payload = {k: v for k, v in user_data.items()}
    payload["exp"] = f"Expires in {args.expires} hours"
    print(json.dumps(payload, indent=2))
    print("\n=== Usage Examples ===")
    print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/datasets')
    print(f'\nIn Python:')
    print(f'headers = {{"Authorization": "Bearer {token}"}}')
    print(f'response = requests.get("http://localhost:8000/datasets", headers=headers)')

if __name__ == "__main__":
    main()
