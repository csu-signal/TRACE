import os
import sys
import socket
import re
import random
import numpy as np
import torch
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Hugging Face Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel

# T = TypeVar("T", bound=BaseInterface)


# class BaseFeature(ABC, Generic[T]):
#     """
#     The base class all features in the demo must implement.
#     """

#     def __init__(self, *args) -> None:
#         self._deps = []
#         self._rev_deps = []
#         self._register_dependencies(args)

#     def _register_dependencies(self, deps: "list[BaseFeature] | tuple"):
#         """
#         Add other features as dependencies which are required
#         to be evaluated before this feature.

#         Arguments:
#         deps -- a list of dependency features
#         """
#         assert len(self._deps) == 0, "Dependencies have already been registered"
#         for d in deps:
#             self._deps.append(d)
#             d._rev_deps.append(self)

#     @abstractmethod
#     def get_output(self, *args, **kwargs) -> T | None:
#         """
#         Return output of the feature. The return type must be the output
#         interface to provide new data and `None` if there is no new data.
#         It is very important that this function does not modify any of the
#         input interfaces because they may be reused for other features.

#         Arguments:
#         args -- list of output interfaces from dependencies in the order
#                 they were registered. Calling `.is_new()` on any of these
#                 elements will return True if the argument has not been
#                 sent before. It is possible that the interface will not
#                 contain any data before the first new data is sent.
#         """
#         raise NotImplementedError

#     def initialize(self):
#         """
#         Initialize feature. This is where all the time/memory
#         heavy initialization should go. Put it here instead of
#         __init__ to avoid wasting resources when extra features
#         exist. This method is guaranteed to be called before
#         the first `get_output` and run on the main thread.
#         """

#     def finalize(self):
#         """
#         Perform any necessary cleanup. This method is guaranteed
#         to be called after the final `get_output` and run on the
#         main thread.
#         """

#     def is_done(self) -> bool:
#         """
#         Return True if the demo should exit. This will
#         always return False if not overridden.
#         """
#         return False

@dataclass
class FrictionMetrics:
    """Metrics for generated friction statements"""
    nll: float
    predictive_entropy: float
    mutual_information: float
    perplexity: float
    conditional_entropy: float

@dataclass
class FrictionOutputInterface:
    """
    Interface for friction generation output in collaborative weight estimation task.

    Attributes:
        friction_statement (str):
            Main friction statement to be displayed/spoken.
            Example: "Are we sure about comparing these blocks without considering their volume?"

        task_state (str):
            Current state of the weight estimation task.
            Hidden from UI but useful for debugging.
            Example: "Red (10g) and Blue blocks compared, Yellow block pending"

        belief_state (str):
            Participants' current beliefs about weights.
            Helps explain friction but may not need display.
            Example: "P1 believes yellow is heaviest, P2 uncertain about blue"

        rationale (str):
            Reasoning behind the friction intervention.
            Could be shown as tooltip/explanation.
            Example: "Participants are making assumptions without evidence"

        metrics (Optional[FrictionMetrics]):
            Model's generation metrics including confidence.
            Useful for debugging and demo insights.
    """

    friction_statement: str
    task_state: str
    belief_state: str
    rationale: str
    raw_generation: str

    metrics: Optional[FrictionMetrics] = None

    def to_dict(self):
        return asdict(self)  # Converts the object into a dictionary

class FrictionInference:
    def __init__(self, model_path: str, generation_args=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tags_to_parse = ["friction", "rationale", "t", "b"]
        self.model_path = model_path
        # Set default generation args if none provided
        if generation_args is None:
            self.generation_args = {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "return_dict_in_generate": True,
                "output_scores": True
            }
        else:
            self.generation_args = generation_args
        print("Loading base model...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Merge and unload adapter weights
        # print("Merging and unloading adapter weights...",  self.model)
        # self.model = self.model.merge_and_unload()
        self.model = self.model.to(self.device)  # Move the model to the GPU device
        print("showing merged lora model",  self.model)

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.tokenizer.padding_side = 'right'

    def parse_generation(self, text: str) -> Dict[str, str]:
        """Parse generated text to extract components"""
        parsed = {tag: [] for tag in self.tags_to_parse}

        for tag in self.tags_to_parse:
            pattern = f"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, text, re.DOTALL)
            parsed[tag].extend(matches)

        # Handle friction tag specially
        if not parsed["friction"]:
            parsed["friction"] = [self._extract_friction(text)]

        return {k: " ".join(v).strip() for k, v in parsed.items()}

    def _extract_friction(self, text: str) -> str:
        """Extract friction statement when tags are missing"""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        return " ".join(sentences)

    def parse_tags_robust(self, text, tags=None):
        """
        Combined parsing function that handles both tagged friction (<friction>) and
        other formats (*Friction, ### Friction, etc.) but returns results in the
        parse_tags format.
        
        Args:
            text (str): The text to parse.
            tags (list): List of tags to parse. If None, defaults to standard tags.
            
        Returns:
            dict: Dictionary with tag names as keys and lists of extracted content as values.
        """
        import re
        if tags is None:
            tags = ["friction", "rationale", "t", "b"]
        
        # Initialize result in the format of parse_tags_robust
        result = {tag: [] for tag in tags}
        
        # First try parse_tags_robust to get standard tag format
        for tag in tags:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            matches = re.findall(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", text, re.DOTALL)
            if matches:
                result[tag].extend(matches)
        
        # For any tags that weren't found, try the llm_response patterns
        # Only apply this to tags that are empty and are in our supported list
        # This makes sure we don't overwrite any successful tag extractions
        supported_tags = {"friction", "rationale", "t", "b"}
        
        for tag in tags:
            if not result[tag] and tag in supported_tags:
                # Define patterns based on tag type
                if tag == "friction":
                    primary_patterns = [
                        r'### Friction\n(.*?)(?=\n\n|\Z)',
                        r'\*\*Friction:\*\*\n(.*?)(?=\n\n|\Z)', 
                        r'\*\*Friction\*\*\n(.*?)(?=\n\n|\Z)',
                        r'Friction:(.*?)(?=\n\n|\Z)'
                    ]
                    backup_patterns = [
                        r'friction.*?:(.*?)(?=\n\n|\Z)',
                        r'intervention.*?:(.*?)(?=\n\n|\Z)'
                    ]
                elif tag == "rationale":
                    primary_patterns = [
                        r'### Rationale\n(.*?)(?=\n\n|\Z)', 
                        r'\*\*rationale\*\*\n(.*?)(?=\n\n|\Z)',
                        r'\*\*Rationale\*\*\n(.*?)(?=\n\n|\Z)',
                        r'Rationale:(.*?)(?=\n\n|\Z)'
                    ]
                    backup_patterns = [
                        r'rational.*?:(.*?)(?=\n\n|\Z)',
                        r'reasoning.*?:(.*?)(?=\n\n|\Z)',
                        r'reason.*?:(.*?)(?=\n\n|\Z)'
                    ]
                elif tag == "t":  # Task State
                    primary_patterns = [
                        r'### Task[- ]?State\n(.*?)(?=\n\n|\Z)',
                        r'\*\*Task[- ]?State\*\*\n(.*?)(?=\n\n|\Z)',
                        r'Task[- ]?State:(.*?)(?=\n\n|\Z)'
                    ]
                    backup_patterns = [
                        r'task.*?state.*?:(.*?)(?=\n\n|\Z)',
                        r'current.*?task.*?:(.*?)(?=\n\n|\Z)'
                    ]
                elif tag == "b":  # Belief State
                    primary_patterns = [
                        r'### Belief[- ]?State\n(.*?)(?=\n\n|\Z)',
                        r'\*\*Belief[- ]?State\*\*\n(.*?)(?=\n\n|\Z)',
                        r'Belief[- ]?State:(.*?)(?=\n\n|\Z)'
                    ]
                    backup_patterns = [
                        r'belief.*?state.*?:(.*?)(?=\n\n|\Z)',
                        r'beliefs.*?:(.*?)(?=\n\n|\Z)',
                        r'state.*?:(.*?)(?=\n\n|\Z)'
                    ]
                else:
                    # Skip other tags
                    continue
                
                # Try primary patterns first
                content = None
                for pattern in primary_patterns:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        content = re.sub(r'^\s*\*\s*', '', content, flags=re.MULTILINE)
                        break
                
                # Try backup patterns if primary fails
                if not content:
                    for pattern in backup_patterns:
                        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                        if match:
                            content = match.group(1).strip()
                            content = re.sub(r'^\s*\*\s*', '', content, flags=re.MULTILINE)
                            break
                
                # If we found content, add it to the result
                if content:
                    result[tag].append(content)
        
        # Check if we're still missing any tags and try chunk-based approach as last resort
        missing_tags = [tag for tag in tags if not result[tag] and tag in supported_tags]
        if missing_tags:
            # Split text into chunks by double newlines
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            
            if chunks:
                # Get the last chunk for friction if it's missing
                if "friction" in missing_tags and chunks:
                    result["friction"].append(chunks[-1])
                
                # Get the first chunk for task state if it's missing
                if "t" in missing_tags and len(chunks) > 0:
                    result["t"].append(chunks[0])
                
                # Get the second chunk for belief state if it's missing
                if "b" in missing_tags and len(chunks) > 1:
                    result["b"].append(chunks[1])
                
                # Get a middle chunk for rationale if it's missing
                if "rationale" in missing_tags and len(chunks) > 2:
                    result["rationale"].append(chunks[len(chunks)//2])
        
        return result
    
    
    def handle_friction_logic(self, text):
        '''
        This function processes a text string to extract or construct a "friction" snippet by:
    
        Returning the text following a <friction> tag if present, unless a closing </friction> tag is found.
        If no <friction> tags exist, it constructs a snippet by extracting the first, second-to-last, 
        and last sentences if there are at least three sentences; otherwise, it returns all available sentences.
        
        '''
        if "<friction>" not in text and "</friction>" not in text:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
            if len(sentences) >= 3:
                return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
            elif sentences:
                return " ".join(sentences)
            else:
                return ""
        elif "<friction>" in text and "</friction>" not in text:
            friction_start = text.find("<friction>") + len("<friction>")
            return text[friction_start:].strip()
        else:
            return ""  # Friction is complete, no need to handle further

    def compute_metrics(self, output_ids: torch.Tensor, scores: List[torch.Tensor], prompt_length: int) -> FrictionMetrics:
        """Compute generation metrics (fixed device handling)"""
        with torch.no_grad():
            # Ensure all tensors are on the same device
            logits = torch.stack(scores, dim=0).to(self.device)
            output_ids = output_ids.to(self.device)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Get generated token probabilities
            token_ids = output_ids[prompt_length:]
            probs = probs[:, 0, :]  # Take first sequence
            token_probs = probs[torch.arange(len(token_ids), device=self.device), token_ids]

            # Calculate metrics
            nll = -torch.sum(torch.log(token_probs)) / len(token_ids)
            predictive_entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            conditional_entropy = -torch.mean(torch.log(token_probs))
            mutual_information = max(predictive_entropy - conditional_entropy, 0.0)
            perplexity = torch.exp(nll)

            return FrictionMetrics(
                nll=nll.item(),
                predictive_entropy=predictive_entropy.item(),
                mutual_information=mutual_information.item(),
                perplexity=perplexity.item(),
                conditional_entropy=conditional_entropy.item()
            )

    def generate_friction(self, dialogue_history: str, seed: int = 42) -> dict:
        # Format prompt
        torch.manual_seed(seed)
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

        # Final formatted prompt for the LLM
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt_rm}

            {friction_definition_game_definition_prompt_rm}

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            {dialogue_history}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            ### Assistant:"""

        # Generate response
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=200,
            #     temperature=0.7,
            #     do_sample=True,
            #     top_p=0.9,
            #     return_dict_in_generate=True,
            #     output_scores=True
            # )
            outputs = self.model.generate(
            **inputs,
            **self.generation_args
        )
        



            generated_ids = outputs.sequences[0]
            generated_ids = generated_ids.to(self.device)
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            scores = outputs.scores  # Already on correct device, don't move to list

            metrics = self.compute_metrics(generated_ids, scores, inputs['input_ids'].shape[1]) # probably not needed
            parsed = self.parse_generation(generated_text)
            #print("\nGenerated text:", generated_text)  # Debug print

            # Parse components: if not FAAF agent, carry on with previous parsing code; else, use more robust parsing for FAAF (FAAF does not like to generate <friction> tokens, so needs different parsing logic)
            if self.model_path.split("/")[-1] != 'intervention_agent':
                
                task_state = self._extract_tag(generated_text, "t")
                belief_state = self._extract_tag(generated_text, "b")
                rationale = self._extract_tag(generated_text, "rationale")
                friction = self._extract_tag(generated_text, "friction")
         

            else: # this will run if FAAF agent is being used since the last part after splitting is intervention_agent

                tags_for_parsing = ["friction", "rationale", "t", "b"]  
                parsed_frictive_states_and_friction = self.parse_tags_robust(generated_text, tags_for_parsing)
                friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('friction', []))
                if not friction_intervention:
                    friction_intervention = self.handle_friction_logic(generated_text)
                friction = friction_intervention
                rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
                belief_state = ' '.join(parsed_frictive_states_and_friction.get('b', []))  # Note: Using 'b' not 'belief_state'
                task_state = ' '.join(parsed_frictive_states_and_friction.get('t', []))    # Note: Using 't' not 'task_state'

            print("task state:", task_state)
            print("belief state:", belief_state)
            print("rationale:", rationale)
            print("friction:", friction)
            print("metrics", metrics, "\n")


            return FrictionOutputInterface(
                friction_statement=friction,
                task_state=task_state,
                belief_state=belief_state,
                rationale=rationale,
                metrics=metrics,
                raw_generation=generated_text)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None

    def _extract_tag(self, text: str, tag: str) -> str:
        import re
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else ""

# def start_server(friction_detector: FrictionInference):
#     HOST = '129.82.138.15'  # Standard loopback interface address (localhost)
#     PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         print(f"Server listening on {HOST}:{PORT}")

#         while True:
#             conn, addr = s.accept()
#             with conn:
#                 print(f"Connected by {addr}")
#                 while True:
#                     try:
#                         data = conn.recv(2048)
#                         if not data:
#                             break
#                         print("Received Data Length:" + str(len(data)))
#                         transcriptions = data.decode()
#                         print(f"Transcriptions:\n{transcriptions}")
#                         print("\nGenerating friction for dialogue...")
#                         result = friction_detector.generate_friction(transcriptions)
#                         returnString = ''
#                         if result is not None:
#                             if result.friction_statement != '':
#                                 returnString += "Friction: " + result.friction_statement
#                                 if result.rationale != '':
#                                     returnString += "r*Rationale" + result.rationale
#                             else:
#                                 conn.sendall(str.encode("No Friction", 'utf-8'))
#                                 break
#                             returnString = returnString.replace("’","'")
#                             conn.sendall(str.encode(returnString, 'utf-8'))
#                         else:
#                             conn.sendall(str.encode("No Friction", 'utf-8'))
#                     except ConnectionResetError as e:
#                         print(f"Connection with {addr} was reset: {e}")
#                         break
#                     except Exception as e:
#                         print(f"An error occurred: {e}")
#                         break

def start_local(transcriptions, friction_detector: FrictionInference):
    print(f"Transcriptions:\n{transcriptions}")
    print("\nGenerating friction for dialogue...")
    result = friction_detector.generate_friction(transcriptions)
    returnString = ''
    if result is not None:
        if result.friction_statement != '':
            returnString += "Friction: " + result.friction_statement
            if result.rationale != '':
                returnString += "r*Rationale" + result.rationale
        else:
            return("No Friction", 'utf-8')
        returnString = returnString.replace("’","'")
        return(returnString, 'utf-8')
    else:
        return("No Friction", 'utf-8')


# if __name__ == "__main__":
#     print("Initializing friction detector...")
#     friction_detector = FrictionInference("Abhijnan/friction_sft_allsamples_weights_instruct") #this is the lora model id on huggingface (SFT model)
    # friction_detector = FrictionInference("Abhijnan/dpo_friction_run_with_69ksamples") #this is the dpo model
    # start_server(FrictionInference("Abhijnan/friction_sft_allsamples_weights_instruct")) #this is the lora model id on huggingface (SFT model)