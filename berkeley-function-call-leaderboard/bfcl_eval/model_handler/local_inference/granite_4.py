import json
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)
from overrides import override


class Granite4FCHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = True
        self.model_name_huggingface = model_name.replace("-FC", "")

    @override
    def decode_ast(self, result, language="Python"):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"?
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            return []
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            return []
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """chat template:
        {%- set tools_system_message_prefix = 'You are a helpful assistant with access to the following tools. You may call one or more tools to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>'  %}
        {%- set tools_system_message_suffix = '\n</tools>\n\nFor each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.' %}
        {%- set documents_system_message_prefix = 'You are a helpful assistant with access to the following documents. You may use one or more documents to assist with the user query.\n\nYou are given a list of documents within <documents></documents> XML tags:\n<documents>' %}
        {%- set documents_system_message_suffix = '\n</documents>\n\nWrite the response to the user\'s input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.' %}
        {%- if available_tools is defined and available_tools %}
            {%- set tools = available_tools %}
        {%- endif %}
        {%- set ns = namespace(tools_system_message=tools_system_message_prefix,
                            documents_system_message=documents_system_message_prefix,
                            system_message='',
                            last_query_index=messages|length - 1) %}
        {%- if tools %}
            {%- for tool in tools %}
                {%- set ns.tools_system_message = ns.tools_system_message + '\n' + (tool | tojson) %}
            {%- endfor %}
            {%- set ns.tools_system_message = ns.tools_system_message + tools_system_message_suffix %}
        {%- else %}
            {%- set ns.tools_system_message = '' %}
        {%- endif %}
        {%- if documents %}
            {%- for document in documents %}
                {%- set ns.documents_system_message = ns.documents_system_message + '\n' + (document | tojson) %}
            {%- endfor %}
            {%- set ns.documents_system_message = ns.documents_system_message + documents_system_message_suffix %}
        {%- else %}
            {%- set ns.documents_system_message = '' %}
        {%- endif %}
        {%- if messages[0].role == 'system' %}
            {%- if messages[0].content is string %}
                {%- set ns.system_message = messages[0].content %}
            {%- elif messages[0].content is iterable %}
                {%- for entry in messages[0].content %}
                    {%- if entry.type== 'text' %}
                        {%- if ns.system_message != '' %}
                            {%- set ns.system_message = ns.system_message + '\n' %}
                        {%- endif %}
                        {%- set ns.system_message = ns.system_message + entry.text %}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {%- if tools and documents %}
                {%- set ns.system_message = ns.system_message + '\n\n' +  ns.tools_system_message + '\n\n' + ns.documents_system_message %}
            {%- elif tools %}
                {%- set ns.system_message = ns.system_message + '\n\n' + ns.tools_system_message %}
            {%- elif documents %}
                {%- set ns.system_message = ns.system_message + '\n\n' + ns.documents_system_message %}
            {%- endif %}
        {%- else %}
            {%- if tools and documents %}
                {%- set ns.system_message = ns.tools_system_message + '\n\n' + ns.documents_system_message  %}
            {%- elif tools %}
                {%- set ns.system_message = ns.tools_system_message %}
            {%- elif documents %}
                {%- set ns.system_message = ns.documents_system_message %}
            {%- endif %}
        {%- endif %}
        {%- if ns.system_message %}
            {{- '<|start_of_role|>system<|end_of_role|>' + ns.system_message + '<|end_of_text|>\n' }}
        {%- endif %}
        {%- for message in messages|reverse %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if message.role == 'user' %}
                {%- set content = namespace(val='') %}
                {%- if message.content is string %}
                    {%- set content.val = message.content %}
                {%- else %}
                    {%- if message.content is iterable %}
                        {%- for entry in message.content %}
                            {%- if entry.type== 'text' %}
                                {%- if content.val != '' %}
                                    {%- set content.val = content.val + '\n' %}
                                {%- endif %}
                                {%- set content.val = content.val + entry.text %}
                            {%- endif %}
                        {%- endfor %}
                    {%- endif %}
                {%- endif %}
                {%-if not(content.val.startswith('<tool_response>') and content.val.endswith('</tool_response>')) %}
                    {%- set ns.last_query_index = index %}
                    {% break %}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- set content = namespace(val='') %}
            {%- if message.content is string %}
                {%- set content.val = message.content %}
            {%- else %}
                {%- if message.content is iterable %}
                    {%- for entry in message.content %}
                        {%- if entry.type== 'text' %}
                            {%- if content.val != '' %}
                                {%- set content.val = content.val + '\n' %}
                            {%- endif %}
                            {%- set content.val = content.val + entry.text %}
                        {%- endif %}
                    {%- endfor %}
                {%- endif %}
            {%- endif %}
            {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}
                {{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val + '<|end_of_text|>\n' }}
            {%- elif message.role == 'assistant' %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content.val %}
                        {%- set reasoning_content = content.val.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content.val = content.val.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if reasoning_content %}
                        {{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.val.lstrip('\n') }}
                    {%- else %}
                        {{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val }}
                    {%- endif %}
                {%- else %}
                    {{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val }}
                {%- endif %}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if (loop.first and content.val) or (not loop.first) %}
                            {{- '\n' }}
                        {%- endif %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
                        {%- if tool_call.arguments is string %}
                            {{- tool_call.arguments }}
                        {%- else %}
                            {{- tool_call.arguments | tojson }}
                        {%- endif %}
                        {{- '}\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|end_of_text|>\n' }}
            {%- elif message.role == 'tool' %}
                {%- if loop.first or (messages[loop.index0 - 1].role != 'tool') %}
                    {{- '<|start_of_role|>user<|end_of_role|>' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content.val }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}
                    {{- '<|end_of_text|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|start_of_role|>assistant<|end_of_role|>' }}
            {%- if thinking is defined and thinking is true %}
                {{- '<think>\n' }}
            {%- endif %}
        {%- endif %}
        """
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|start_of_role|>system<|end_of_role|>"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "You are a helpful assistant with access to the following tools. You may call one or more tools to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.<|end_of_text|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|start_of_role|>system<|end_of_role|>{messages[0]['content']}<|end_of_text|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|start_of_role|>{role}<|end_of_role|><think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|start_of_role|>{role}<|end_of_role|>{content}"
                else:
                    formatted_prompt += f"<|start_of_role|>{role}<|end_of_role|>{content}"
                    
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|end_of_text|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|start_of_role|>user<|end_of_role|>"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|end_of_text|>\n"

        formatted_prompt += "<|start_of_role|>assistant<|end_of_role|>"
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        # FC models use its own system prompt, so no need to add any message

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }

        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }
            
        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
            except Exception as e:
                pass
            result.append(match)
        return result
