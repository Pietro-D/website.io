---
layout: project
title: A Pipeline for Solving Hard Math Problems
description: A pipeline for solving challenging math problems using <b> GPT-OSS-120b </b>, <b> Tool Integrated Reasoning </b> and <b> Self-Consistency</b>.
image: /assets/images/mathLLM.png
github_link: https://github.com/Pietro-D/A-Pipeline-for-Solving-Hard-Math-Problems-using-GPT-OSS
tools: [Python, LLMs, Vllm, OpenAI]
permalink: /projects/mathLLM/
featured: true
---
## Table of Contents
[Introduction](#itroduction)<br>
[Utility Functions](#utility-functions)<br>
[System Prompts](#system-prompts)<br>
[Defining the Server-Client Infrastructure](#defining-the-server-client-infrastructure)<br>
[Python Tool](#python-tool)<br>
[Inference](#inference)<br>
[Evaluation](#evaluation)
## Introduction 
Despite the significant advancements in Large Language Models (LLMs) in
recent years, complex reasoning tasks such as advanced mathematics and
programming remain a major challenge, even for the most powerful models.
Standard benchmarks such as <b>MATH</b>, <b>GSM8K</b> and
<b>AIME2025</b> show many models achieving high accuracy scores; however, performance drops significantly when tackling truly difficult problems.
In particular, LLMs still struggle with problems at the level of the <b>
International Mathematical Olympiad (IMO)</b>. To date, only two models
have reportedly reached a gold-medal standard at the IMO: an advanced
version of [Gemini](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) 
developed by DeepMind and a model from [OpenAI](https://x.com/OpenAI/status/1946594928945148246).
Both are closed-source, highlighting the significant gap between open and proprietary models

The goal of this project is to evaluate the real capabilities of
open-source models by developing an inference pipeline specifically
designed to solve hard mathematical problems. For this purpose, I selected <b>GPT-OSS-120B</b>, a model with strong reasoning abilities 
released by OpenAI. One of its key advantages is the ability to integrate reasoning with external tools such as Python and web browser.

The strategy I decided to implement is the following one: for each problem,
the model performs <b>8 parallel generations</b>, each initialized with a different system prompt to encourage diversity. 
Each generation is treated as a <b> continuous refinement process 
augmented with tool execution </b>, also known as <b>
Tool Integrated Reasoning </b>.
This tecniques has been shown to significantly improve LLM performance 
on complex reasoning tasks [(Heng Lin, Zhongwen Xu)](https://arxiv.org/abs/2508.19201).

The process can be summarized as follows:
<ol>
<li>Generate a fixed number of tokens (e.g 2048)</li>
<li>If python code is present, execute it and extract the results.</li>
<li>Append both the execution results and the model's response to the conversation
context.</li>
<li>Repeat the process until an answer enclosed in
```\\boxed{}```
is produced.</li>
</ol>
{: .stylish-list}
This iterative approach allows the model to adjust its reasoning path based
on the results of code execution.
Once all generations are complete, the final answer is selected using
a <b>majority vote</b> across the 8 solutions.
To improve efficiency and reduce unnecessary computation, I also
implemented a simple early stopping mechanism that terminates all
generations as soon as the same answer is produced three times.
For simplicity, this project currently focuses on problems
with integer solutions between 0 and 99.999, but the approach can be
easily extended.


## Utility Functions

First, I defined a function <b>extract_boxed_text</b>, which searches
for a ```\\boxed{}``` element in a string and
returns its content. If multiple boxed elements are found, the function returns the last one.

{% highlight python %}
def extract_boxed_text(text: str) -> str:
    """Extract text inside \\boxed{} from LaTeX-formatted text"""
    import re

    pattern: str = r"oxed{(.*?)}"
    matches: list[str] = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""
{% endhighlight %}
Next, I implemented a function to check whether the string extracted from
the boxed element is a valid answer. As discussed earlier
I took into account only problems with integer solutions in the range [0,99.999]. However, this function 
can be easily extended to handle other types of solutions, such as
floating pointing numbers.

{% highlight python %}
def is_valid_answer_string(text: str) -> bool:
    try:
        if int(text) == float(text):
            if 0 <= int(text) <= 99_999:
                return True
    except Exception:
        pass
    return False
{% endhighlight %}

To implement the voting mechanism, I defined a global dictionary that keeps track of the answers provided by the model for each question.
Each problem is assumed to have a unique id.
The function <b>vote_answer</b> updates the global dictionary and marks
a question as completed if the model returns the same answer three times.
If not, the system waits for all 8 generations to complete before selecting the final answer.
{% highlight python %}
completed_question_ids: set[str] = set()
question_id_to_counter: dict[str, Counter] = {"": Counter()}

def vote_answer(question_id: str) -> int | None:
    # reads counter from global
    counter = question_id_to_counter[question_id]
    most_common = counter.most_common(1)[0]
    answer, count = most_common
    
    # Early stop if we have 3+ votes for same answer
    if count >= 3:
        completed_question_ids.add(question_id)
        return answer
    return answer
{% endhighlight %}

## System Prompts
The following are the 8 system prompts used in the project.
Each prompt explicitly encourages the model to utilize Python for
intermediate computations, helping it reason more accurately and efficiently.
{% highlight python %}
SYSTEM_PROMPTS = [
    """You will solve this problem step-by-step using the python tool for ALL calculations, intermediate steps, 
    and verification. Break down complex problems into smaller computational steps. Use the tool to: 
    (1) parse and validate input data, (2) perform all arithmetic operations, (3) implement algorithms iteratively, 
    (4) verify intermediate results, and (5) cross-check your final answer. Think through the problem logically, 
    then execute each step computationally. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
    
    """Approach this problem by integrating reasoning with computation. 
    First, analyze the problem structure and develop a solution strategy. Then use the python tool to: 
    (1) test your approach on simple cases, (2) implement helper functions for complex operations, 
    (3) perform all numerical computations, (4) validate edge cases, and (5) verify your logic with 
    multiple test scenarios. Use computational exploration to refine your understanding. 
    Let calculations guide your reasoning process. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
    
    """Solve this problem using iterative tool-assisted reasoning. Use the python tool as an extension of your 
    thinking process to: (1) explore the problem space computationally, (2) test hypotheses with concrete calculations,
    (3) implement and debug solution algorithms step-by-step, (4) perform intermediate verifications to catch errors early,
    and (5) conduct sensitivity analysis on your results. Don't just calculate—use the tool to think through the problem, 
    discover patterns, and validate your reasoning at each stage. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
    
    """You will solve this problem by deeply integrating computational tools with analytical reasoning. 
    Use the python tool proactively throughout your solution process to: (1) computationally verify each logical step, 
    (2) explore multiple solution approaches empirically, (3) implement sanity checks and boundary condition tests, 
    (4) simulate or enumerate scenarios when beneficial, and (5) trace through your algorithm with concrete examples. 
    Make your reasoning explicit, then validate it computationally. Use the tool not just for final calculations, 
    but as an integral part of your problem-solving process. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",

  """You will solve this problem step-by-step using the python tool for ALL calculations, intermediate steps, 
    and verification. Break down complex problems into smaller computational steps. Use the tool to: 
    (1) parse and validate input data, (2) perform all arithmetic operations, (3) implement algorithms iteratively, 
    (4) verify intermediate results, and (5) cross-check your final answer. Think through the problem logically, 
    then execute each step computationally. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
    
    """Approach this problem by integrating reasoning with computation. 
    First, analyze the problem structure and develop a solution strategy. Then use the python tool to: 
    (1) test your approach on simple cases, (2) implement helper functions for complex operations, 
    (3) perform all numerical computations, (4) validate edge cases, and (5) verify your logic with 
    multiple test scenarios. Use computational exploration to refine your understanding. 
    Let calculations guide your reasoning process. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
    
    """Solve this problem using iterative tool-assisted reasoning. Use the python tool as an extension of your 
    thinking process to: (1) explore the problem space computationally, (2) test hypotheses with concrete calculations,
    (3) implement and debug solution algorithms step-by-step, (4) perform intermediate verifications to catch errors early,
    and (5) conduct sensitivity analysis on your results. Don't just calculate—use the tool to think through the problem, 
    discover patterns, and validate your reasoning at each stage. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
    
    """You will solve this problem by deeply integrating computational tools with analytical reasoning. 
    Use the python tool proactively throughout your solution process to: (1) computationally verify each logical step, 
    (2) explore multiple solution approaches empirically, (3) implement sanity checks and boundary condition tests, 
    (4) simulate or enumerate scenarios when beneficial, and (5) trace through your algorithm with concrete examples. 
    Make your reasoning explicit, then validate it computationally. Use the tool not just for final calculations, 
    but as an integral part of your problem-solving process. Return the final answer in \\boxed{}. 
    The answer is expected to be an integer between 0 and 99999 inclusive.""",
]
{% endhighlight %}

## Defining the Server-Client Infrastructure
Now that the utility functions are defined and the system prompts are ready, the next step is to serve the model using vLLM.
{% highlight python %}
with open("a-vllm.log", "w") as f:
    f.write("")

def start_vllm_server() -> subprocess.Popen[bytes]:
    """Start vLLM server in the background"""
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#troubleshooting
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = (
        "/kaggle/working/tiktoken_encodings"
    )

    sequence_length = 65_536

    command: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/kaggle/input/gpt-oss-120b/transformers/default/1",
        "--served-model-name",
        "vllm-model",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.96",  # any higher may not have enough for graph capture
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--max-model-len",
        f"{sequence_length}",
        "--load-format",
        "fastsafetensors",
    ]

    # Start the process in the background
    with open("/kaggle/working/a-vllm.log", "w") as logfile:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )

    print("Logs: /kaggle/working/a-vllm.log")
    return process

# Start the server
vllm_process: subprocess.Popen[bytes] = start_vllm_server()
{% endhighlight %}
A few key points about the setup:<br>
1) The model is loaded in <b>fastsafetensors</b> format, which
significantly reduces loading time from disk.<br>
2) Since each question requires 8 parallel candidate solutions,
the parameter ```--max_num_seqs``` is set to 8.<br>
3) The maximum context lenght of the model is set to 65.536 tokens, which
provides a good balance between memory usage and accuracy.<br>

Once the server is running, the next step is to instantiate the client
using the OpenAI library.
{% highlight python %}
# Point the client to your local vLLM server
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
os.environ["OPENAI_API_KEY"] = "sk-local"  # any non-empty string

client: OpenAI = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)
{% endhighlight %}
## Python Tool

For the implementation of the Python tool, I adapted code from [gpt-oss](https://github.com/openai/gpt-oss).
The idea is to execute generated Python code using a local Jupyter kernel.
To ensure robustness and efficiency, the tool includes a time manager that
limits the execution of each code block to <b>30 seconds</b>,
which is a reasonable amount of time for this type of task.
Code that takes significantly longer to run typically contains errors
or is not relevant to solving the problem.
{% highlight python %}

class LocalJupyterSession:
    """Stateful helper that proxies execution through a local Jupyter kernel."""

    def __init__(
        self,
        connection_file: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "backend requires the jupyter_client package to be installed."
            ) from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(
                    f"Cannot find Jupyter connection file at '{connection_path}'."
                )
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            # Ensure the connection is ready before executing.
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            km = KernelManager()
            km.start_kernel()
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""

        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for Jupyter kernel output.") from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain the shell channel to capture final execution status.
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Timed out waiting for Jupyter kernel execution reply."
                ) from exc

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            if stdout:
                stdout = f"{stdout.rstrip()}\n{stderr}"
            else:
                stdout = stderr

        if not stdout.strip():
            stdout = (
                "[WARN] No output available. Use print() to output anything to stdout to "
                "receive the output"
            )

        return stdout

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()

class PythonTool():
    def __init__(
        self,
        name: str = "python",
        *,
        local_jupyter_connection_file: str | None = None,
        local_jupyter_timeout: float = 60.0,
    ):
        assert name == "python"

        self._local_jupyter_connection_file = (
            local_jupyter_connection_file
            or os.environ.get("PYTHON_LOCAL_JUPYTER_CONNECTION_FILE")
        )
        self._local_jupyter_timeout = local_jupyter_timeout

        self._execution_lock = threading.Lock()
        self._jupyter_session = LocalJupyterSession(
            connection_file=self._local_jupyter_connection_file,
            timeout=self._local_jupyter_timeout,
        )

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is UNKNOWN. Depends on the cluster.
            """.strip()

       

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(), description=self.instruction, tools=[]
        )

    def _make_response(
        self,
        output: str,
        channel: str | None = None,
    ) -> Message:
        content = TextContent(text=output)
        return self.make_response(content=content, channel=channel)

    def make_response(
        self,
        content: Content,
        *,
        metadata: dict[str, Any] | None = None,
        author: Author | None = None,
        channel: str | None = None,
    ) -> Message:
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=f"{tool_name}")

        message = Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

        if channel:
            message = message.with_channel(channel)

        return message

    def _process(self, message: Message):
        script = message.content[0].text
        channel = message.channel

        
        assert self._jupyter_session is not None
        lock = self._execution_lock
        if lock is not None:
             with lock:
                try:
                    output = self._jupyter_session.execute(script)
                except TimeoutError as exc:
                    output = f"[ERROR] {exc}"
        else:
            try:
                output = self._jupyter_session.execute(script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
        
        yield self._make_response(output, channel=channel)

    def close(self) -> None:
        if self._jupyter_session is not None:
            self._jupyter_session.close()

    def __del__(self) -> None:
        self.close()
{% endhighlight %}
## Inference
We can now define the function that represents the core of the pipeline.
The ```generate_solution``` function performs a single generation, given
as input the problem and the system prompt to be used.

{% highlight python %}
def generate_solution(
    question_text: str, question_id: str = "", solution_index: int = 0, system_prompt: str = ""
) -> str:
    if question_id in completed_question_ids:
        return ""
    
    python_tool = PythonTool()
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids: list[int] = list(harmony_encoding.stop_tokens_for_assistant_actions())
    
    try:
        # Build system message with TIR support
        system_content_obj = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort.HIGH)
            .with_tools(python_tool.tool_config)
        )
        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content_obj)
        developer_message = Message.from_role_and_content(Role.DEVELOPER, system_prompt)
        user_message = Message.from_role_and_content(Role.USER, question_text)
        
        messages = [system_message, developer_message, user_message]
        
       
     
        total_response = ""
        total_tokens = 0
        
        while True: 
            
            # Check global completion
            if question_id in completed_question_ids:
                break

            convo = Conversation.from_messages(messages)
            prompt_ids: list[int] = harmony_encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT
            )
            
            response = client.completions.create(
                model="vllm-model",
                prompt=prompt_ids,
                max_tokens=2048,
                temperature=1.0,
                stream=False,
                extra_body=dict(
                    min_p=0.02,
                    stop_token_ids=stop_token_ids,
                    return_token_ids=True,
                ),
            )
            
            response_ids = getattr(response.choices[0], "token_ids", [])
            text_response = response.choices[0].text or ""
            finish_reason = response.choices[0].finish_reason
            total_response += text_response
            total_tokens += len(response_ids)

            if total_tokens >= 65_536:
                print(f"[{question_id}:{solution_index}] Token limit reached")
                break
            
            if not response_ids:
                print(f"[{question_id}:{solution_index}] No tokens generated, breaking")
                break
            
            # Parse messages FIRST
            new_messages = harmony_encoding.parse_messages_from_completion_tokens(
                response_ids, Role.ASSISTANT
            )
            messages.extend(new_messages)
            last_message = new_messages[-1]

            # Handle tool calls BEFORE checking answer
            if last_message.recipient == "python":
                response_messages = python_tool._process(last_message)
                messages.extend(response_messages)
                continue

            # NOW check for valid answer
            boxed_text = extract_boxed_text(text_response)
            if is_valid_answer_string(boxed_text):
                print(f"[{question_id}:{solution_index}] Valid answer found: {boxed_text}")
                break
            
            # Nudge towards completion
            if total_tokens >= 10_000:
                finish_msg = Message.from_role_and_content(
                    Role.USER,
                    "Provide your final answer in \\boxed{} now."
                )
                messages.append(finish_msg)
                continue  # Go to next iteration

            if finish_reason == 'stop':
                skip_msg = Message.from_role_and_content(
                    Role.USER,
                    "Provide your final answer in \\boxed{} now."
                )
                messages.append(skip_msg)
        
        # Final extraction
        boxed_text = extract_boxed_text(total_response)
        
        # Save results
        path = f"solutions/{question_id}/{solution_index}.txt"
        with open(path, "w") as f:
            f.write(total_response)
        
        # Record answer for voting
        if is_valid_answer_string(boxed_text):
            question_id_to_counter[question_id][int(boxed_text)] += 1
            vote_answer(question_id)
            print(f"[{question_id}:{solution_index}] Final answer: {boxed_text}")
        else:
            print(f"[{question_id}:{solution_index}] No valid answer found")
        
        return boxed_text
        
    except Exception as e:
        print(f"[{question_id}:{solution_index}] Error: {e}")
        return ""
    
    finally:
        try:
            python_tool.close()
        except Exception:
            pass
{% endhighlight %}

As described in the introduction, the process starts from a default
prompt containing the system message and the problem to be solved.
The model then generates the first 2048 tokens. If Python code is detected,
it is executed and the results are extracted. Both the model's
response at the current step and the execution results are then
appended to the conversation context.<br>
<br>
This allows the model, at the next iteration, to continue generating its response while taking into account the results of the code
executed in the previous step.<br>
The loop terminates when one of the following conditions is met:<br>
<ol>
<li>The total number of generated tokens exceeds <b>65.536</b></li>
<li>The question has been marked as completed by the voting mechanism.</li>
<li> A valid answer is found</li>
</ol>
{: .stylish-list}
Finally, combining all components together, we define the ```solve```
function. Given a question and its corresponding ID, this function orchestrates the full pipeline and returns the final answer

{% highlight python %}
def solve(question_text: str, question_id: str = "") -> int:
    await_client()
    
    print(f"processing {question_id}")
    os.makedirs(f"solutions/{question_id}", exist_ok=True)
    question_id_to_counter[question_id] = Counter()
    completed_question_ids.discard(question_id)  # just in case question_id collides
    
    num_generations = 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # run in parallel with different system prompts
        results = executor.map(
            generate_solution,
            [question_text] * num_generations,
            [question_id] * num_generations,
            list(range(num_generations)),
            SYSTEM_PROMPTS
        )
        list(results)
    
    
    final_answer = vote_answer(question_id)
    
    return final_answer
{% endhighlight %}
## Evaluation

The pipeline was evaluated on 35 IMO-level problems. 
The problems were extracted from
the [olympiads-ref-base dataset](https://huggingface.co/datasets/AI-MO/olympiads-ref-base).
Specifically, I selected the first 35 problems with integer solutions.
The model correctly solved <b>26 out of 35 problems (74,28%)</b> problems,
which is a strong result considering the difficulty of the tasks and the
fact that no fine-tuning tecniques where applied.<br>
The full source code, along with the evaluation dataset and experimental setup, is available on my GitHub repository.
