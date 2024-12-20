from huggingface_hub import InferenceClient

api_key = "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"
client = InferenceClient(api_key=api_key)

def create_message(user_symptoms):
    return [
        {"role": "system", "content": '''Assume you are a doctor with extensive knowledge of disease and its cure you will be given 
                                     a text with some symptoms and you have analyze that symptoms and predict the what is diseases
                                     user might have and recommend a short remedies for it in keywords not a sentence'''}, 
        {"role": "user", "content": user_symptoms},
    ]

def predict_disease(symptoms):
    try:
        message = create_message(symptoms)
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=message,
            max_tokens=512
        )
        response = response.choices[0].message['content']
        return response
    except Exception as e:
        raise Exception(f"Error in disease prediction: {str(e)}")