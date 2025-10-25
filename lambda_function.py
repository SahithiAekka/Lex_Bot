import boto3
import json
import os

# Initialize AWS service clients
comprehend = boto3.client("comprehend", region_name="us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    try:
        # Get user input - handle both Lex V1 and V2
        if 'inputTranscript' in event:
            user_input = event['inputTranscript']
        else:
            user_input = "Hello"
        
        # Get intent name safely
        try:
            intent_name = event['sessionState']['intent']['name']
        except:
            intent_name = "FallbackIntent"
        
        # Step 1: Analyze message using Comprehend
        entities = comprehend.detect_entities(Text=user_input, LanguageCode="en")["Entities"]
        
        # Step 2: Get resume summary
        resume_summary = os.environ.get("RESUME_SUMMARY", "Cloud Solutions Architect with AWS expertise")
        
        # Step 3: Build Titan prompt
        prompt = f"""You are Sahithi Aekka's AI assistant at conferences and networking events.

Resume summary: {resume_summary}

Question from recruiter/contact: "{user_input}"

Detected entities in their message: {entities}

Provide a professional, friendly, and natural response that highlights relevant experience and skills.

Response:"""
        
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 300,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        # Step 4: Call Bedrock with Titan
        response = bedrock.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps(body)
        )
        
        # Step 5: Parse Titan response
        response_body = json.loads(response["body"].read())
        answer = response_body["results"][0]["outputText"].strip()
        
        print(f"Generated response: {answer}")
        
        # Step 6: Return in Lex V2 format
        return {
            "sessionState": {
                "dialogAction": {
                    "type": "Close"
                },
                "intent": {
                    "name": intent_name,
                    "state": "Fulfilled"
                }
            },
            "messages": [
                {
                    "contentType": "PlainText",
                    "content": answer
                }
            ]
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return error in Lex V2 format
        return {
            "sessionState": {
                "dialogAction": {
                    "type": "Close"
                },
                "intent": {
                    "name": "FallbackIntent",
                    "state": "Failed"
                }
            },
            "messages": [
                {
                    "contentType": "PlainText",
                    "content": "I'm having trouble processing that. Please try again!"
                }
            ]
        }