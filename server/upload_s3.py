import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 AWS 자격 증명 및 설정 불러오기
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

client = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_DEFAULT_REGION
                      )

file_path = 'data/dino_voice/whisper_result.wav'      # 업로드할 파일 이름 
bucket = 'jurassic-park'           #버켓 주소
key = 'whisper_result.wav' 

async def upload_file_to_s3(file_path, key):
    try:
        # 파일 업로드
        await client.upload_file(file_path, bucket, key)

        # 파일의 퍼블릭 URL 생성
        file_url = f"https://{bucket}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{key}"
        
        return file_url

    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None

file_url = upload_file_to_s3(file_path, key)

if file_url:
    print("File uploaded successfully!")
    print(f"File URL: {file_url}")
else:
    print("File upload failed.")