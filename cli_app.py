from inference import generate_response

if __name__=="__main__":
    context=""
    while(True):
      print("Prompt:")
      prompt=input()
      response,context=generate_response(prompt,context)
      print(f"Response:{response}")
    