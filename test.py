from transformers import pipeline

print("Loading fine-tuned model...")
qa = pipeline("text-classification", model="./model", tokenizer="./model")

question = "advantages of GM plants"
print(f"Question: {question}")

result = qa(question)
label = result[0]["label"]
score = result[0]["score"]

if score < 0.1:
    print("Predicted answer: Sorry, I don't know the answer to that.")
else:
    print(f"\nPredicted answer: {label}")
    print(f"Confidence score: {score:.2f}") 

