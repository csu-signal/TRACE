from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

class FrictionInference:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model using PEFT AutoModel
        print("Loading base model...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Merge and unload adapter weights
        print("Merging and unloading adapter weights...")
        self.model = self.model.merge_and_unload()
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.tokenizer.padding_side = 'right'
        
    def generate_friction(self, dialogue_history: str) -> dict:
        # Format prompt

        system_prompt_rm = (
        "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
        "Your task is to analyze the dialogue history involving three participants and the game details "
        "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
        "Finally, generate a nuanced friction statement in a conversational style based on your analysis.\n\n"
        "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
        "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
        "3. Provide a rationale for why a friction statement is needed. This monologue must be enclosed between the "
        "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
        "- Incorrect assumptions\n"
        "- False beliefs\n"
        "- Rash decisions\n"
        "- Missing evidence.\n\n"
        "4. Generate the friction statement, ensuring it is enclosed between the markers `<friction>` and `</friction>`. "
        "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
        )

        friction_definition_game_definition_prompt_rm = (
        "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
        "Participants can weigh two blocks at a time and know the weight of the red block. "
        "They must deduce the weights of other blocks. "
        "The dialogue history is provided below:"
        )
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in analyzing group dynamics in the Weights game. Generate a friction statement to encourage reevaluation of beliefs.

<|eot_id|><|start_header_id|>user<|end_header_id|>
{dialogue_history}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
### Assistant:"""

        # Generate response
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            print("\nGenerated text:", generated_text)  # Debug print
            
            # Parse components
            task_state = self._extract_tag(generated_text, "t")
            belief_state = self._extract_tag(generated_text, "b")
            rationale = self._extract_tag(generated_text, "rationale")
            friction = self._extract_tag(generated_text, "friction")
            
            return {
                "friction_statement": friction,
                "task_state": task_state,
                "belief_state": belief_state,
                "rationale": rationale,
                "raw_output": generated_text
            }
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None
        
    def _extract_tag(self, text: str, tag: str) -> str:
        import re
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else ""

# Example usage
if __name__ == "__main__":
    # Sample dialogue from Weights game
    dialogue = """
    P1: Let's try weighing the red and blue blocks together.
    P2: Okay, they weigh 20 grams combined.
    P1: Since we know red is 10 grams, blue must also be 10.
    P3: Are we sure? Maybe we should check again.
    P1: No need, it's pretty clear.
    """

    dialogue1 = """
    P1: Let's start by weighing the red block with the blue one.
    P2: Good idea. The scale shows 20 grams together.
    P1: Well, since we know red is 10 grams, blue must also be 10.
    P3: Should we verify that with another combination?
    P1: No need, it's pretty clear.
    P2: Okay, let's try the purple one next.
    P1: We could weigh it with the blue block.
    P3: They show 40 grams together.
    P2: So if blue is 10, purple must be 30.
    P1: That makes sense to me.
    P3: Wait, shouldn't we also check purple with red to be sure?
    P2: We've already figured it out, let's move on to yellow.
    P1: Yeah, we're making good progress.
    """
    
    dialogue2 = """
    P1: Shall we start with weighing red and blue blocks?
    P2: The scale reads 20 grams for both.
    P3: Since red is 10 grams, blue should be 10 too.
    P1: Right. Now let's try blue and purple together.
    P2: I'm getting... 40 grams total.
    P1: Interesting, so purple must be 30 if blue is 10.
    P2: Makes sense. Should we check the yellow one now?
    P3: Wait, maybe we should weigh purple with red first?
    P1: Why? We already know all their weights.
    P2: Yeah, we can calculate everything now.
    P3: I'm not completely sure...
    P1: Trust me, the math adds up perfectly.
    """
    
    dialogue3 = """
    P1: Let's begin by checking red and blue together.
    P2: They weigh exactly 20 grams.
    P3: Red is 10 grams, so blue must be 10 too.
    P1: Perfect. Let's try purple next.
    P2: I'll weigh it with the blue one... 40 grams total.
    P3: Hold on, before we calculate purple's weight-
    P1: It's obviously 30 grams. Blue is 10, so 40 minus 10.
    P2: That seems reasonable.
    P3: But maybe we should verify with different combinations?
    P1: We don't have time for that.
    P2: Yeah, and our logic is solid.
    P3: What if there's something we're missing though?
    P1: The weights are clear enough from what we've measured.
    """

 
    
    print("Initializing friction detector...")
    friction_detector = FrictionInference("Abhijnan/friction_sft_allsamples_weights_instruct")
    
    print("\nGenerating friction for dialogue...")
    result = friction_detector.generate_friction(dialogue1)
    
    if result:
        print("\nFriction Statement:", result["friction_statement"])
        print("Task State:", result["task_state"])
        print("Belief State:", result["belief_state"])
        print("Rationale:", result["rationale"])