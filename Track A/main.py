#!/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional
import pandas as pd
import httpx
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from _types import ToolCall
from logger import init_logger
from utils import (
    print_model_response,
    print_tool_call,
    print_tool_result,
    extract_answer,
    extract_answer_all,
    compute_score,
)

os.environ['AGENT_API_KEY'] = 'sk-XXXXXXXXXXXXX'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

class Environment:
    """
    Responsible for:
    - discovering tool descriptors from FastAPI `/tools`
    - executing tool calls requested by the LLM
    - applying per-scenario context via X-Scenario-Id header
    """

    # server endpoints are different from agent tools. Agent only has access to tools exposed via /tools endpoint 
    endpoint_mapper = {
        "get_all_scenario": "/scenario/all",
        "get_config_data": "/config-data",
        "get_user_plane_data": "/user-plane-data",
        "get_throughput_logs": "/throughput-logs",
        "get_cell_info": "/cell-info",
        "get_gnodeb_location": "/gnodeb-location",
        "get_user_location": "/user-location",
        "get_serving_cell_pci": "/serving-cell-pci",
        "get_serving_cell_rsrp": "/serving-cell-rsrp",
        "get_serving_cell_sinr": "/serving-cell-sinr",
        "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
        "get_neighboring_cells_pci": "/neighboring-cells-pci",
        "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "get_all_cells_pci": "/all-cells-pci",
        "get_available_tools": "/tools",
        "health": "/health",
        "judge_mainlobe_or_not": "/judge_mainlobe",
        "calculate_horizontal_angle": "/calculate_horizontal_angle",
        "calculate_tilt_angle": "/calculate_tilt_angle",
        "calculate_pathloss": "/calculate_pathloss",
        "calculate_overlap_ratio": "/calculate_overlap_ratio",
        "get_kpi_data": "/get_kpi_data",
        "get_mr_data": "/get_mr_data",
        "optimize_antenna_gain": "/optimize_antenna_gain"
    }

    def __init__(self, server_url: str, verbose: bool = False, log_file: Optional[str] = None, timeout: float = 15.0,
                 logger: logging.Logger = None):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout  # in seconds
        self.logger = logger if logger is not None else init_logger()

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if scenario_id:
            headers["X-Scenario-Id"] = scenario_id
        return headers

    def _call_api(
            self,
            function_name: str,
            scenario_id: Optional[str] = None,
            **params: Any,
    ) -> Dict[str, Any]:
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}

        url = f"{self.server_url}{endpoint}"
        headers = self._headers(scenario_id=scenario_id)

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params}")
            data = resp.json()
            return data
        except requests.exceptions.HTTPError:
            # FastAPI error responses often include {"detail": "..."}
            try:
                detail = resp.json().get("detail", str(resp.text))
            except Exception:
                detail = str(resp.text)
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params} -> HTTPError: {detail}")
            return {"error": detail}
        except Exception as e:
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params} -> ERROR: {e}")
            return {"error": str(e)}

    def get_tools(self) -> List[Dict[str, Any]]:
        """Fetch OpenAI-like tool descriptors from /tools."""
        tools = self._call_api("get_available_tools")
        if isinstance(tools, dict) and "error" in tools:
            return []
        if not isinstance(tools, list):
            return []
        return tools

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Fetch all scenarios available."""
        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            return []
        if not isinstance(scenarios, list):
            return []
        return scenarios

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        """
        Execute a single OpenAI tool_call and return a JSON string for the tool message.
        """
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
            return json.dumps(result, ensure_ascii=False)

        except json.JSONDecodeError:
            error_msg = f"Tool parameter parsing failed: {tool_call.function.arguments}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        except Exception as e:
            error_msg = f"Tool invocation execution failed: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg}, ensure_ascii=False)


# ------------------------------------------------------------------------------
# LLM Agent Runner
# ------------------------------------------------------------------------------

class AgentsRunner:
    """
    Owns:
    - OpenAI client
    - solve() loop (tool calling)
    - benchmark() across scenarios and attempts
    """

    def __init__(
            self,
            environment: Environment,
            model_url: str,
            model_name: str,
            model_provider: Optional[str] = None,
            max_tokens: int = 16000,
            max_retries: int = 3,
            max_iterations: int = 20,
            verbose: bool = False,
            logger: logging.Logger = None
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger if logger is not None else init_logger()
        self.running_metrics = {}

        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
        )

    def _call_model(self, messages: List[Dict[str, Any]], functions: List[Dict[str, Any]], **kwargs):
        base_wait_time = 1.0

        call_kwargs = {
            "model": f"{self.model_provider}/{self.model_name}" if self.model_provider else self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**call_kwargs)
                return response.choices[0].message

            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())

                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    if self.verbose:
                        self.logger.info("Non-retriable exception: %s", exc)
                    return None

                if attempt == self.max_retries:
                    if self.verbose:
                        self.logger.info("Final failure after %s attempts: %s", self.max_retries, exc)
                    return None

                wait = base_wait_time * (2 ** (attempt - 1))
                if self.verbose:
                    self.logger.info("Retry %s/%s after %.1fs due to: %s", attempt, self.max_retries, wait, exc)
                time.sleep(wait)

            except Exception as exc:
                if self.verbose:
                    self.logger.info("Unhandled exception: %s", exc)
                return None

        return None

    def run(self, scenario: Dict[str, Any], free_mode: bool = False) -> Dict[str, Any]:
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {})

        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in task.get("options", [])])

        # tools from server
        tool_defs = self.environment.get_tools()
        if not tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}

        question = task.get("description", "") + f"\nOptions:\n{root_causes}"

        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]

        num_tool_calls = 0
        list_tool_calls = []
        status = None
        reason = None
        last_msg = None

        for i in range(self.max_iterations):
            self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1} conversation, calling tools:")

            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                continue

            last_msg = msg
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            # tool calls
            if msg.tool_calls:
                num_tool_calls += len(msg.tool_calls)

                for j, tool_call in enumerate(msg.tool_calls):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)

                    tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)

                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)

                    has_failed = "error" in tool_result
                    list_tool_calls.append(
                        {
                            "function_name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "turn": i + 1,
                            "has_failed": has_failed,
                            "order": j + 1,
                            "results": tool_result
                        }
                    )

            # final answer
            # elif msg.content or msg.reasoning_content:
            elif msg.content:
                status = "solved"
                break

            else:
                status = "unresolved"
                reason = "Unable to answer this question."
                break

        if status is None:
            status = "unresolved"
            reason = "The maximum number of iterations has been reached."

        # Optional final constraint prompt
        if free_mode:
            current_answer = getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content",
                                                                         "") if last_msg else "",
            current_traces = getattr(last_msg, "reasoning_content", "") if last_msg else ""
            agent_answer = extract_answer(current_answer) or extract_answer(current_traces)
            if agent_answer == "":
                self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 2} conversation, answer question:")
                status = "solved"

                if 'Select the most appropriate optimization solution' in question:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This is a single-answer question. Select the most appropriate optimization solution and enclose its number in \\boxed{{}} "
                                f"in the final answer. For example, \\boxed{{C3}} \nPotential root causes:\n{root_causes}\n"
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This is a multiple-answer question. Select two to four possible optimization solutions and enclose their numbers in \\boxed{{}} "
                                f"in the final answer. For example,  \\boxed{{C3|C5}} or \\boxed{{C7|C11}}. \nPotential root causes:\n{root_causes}\n"
                            ),
                        }
                    )


                msg2 = self._call_model(messages, functions=[])
                if msg2 is not None:
                    last_msg = msg2

        return {
            "scenario_id": scenario_id,
            "num_iterations": (i + 1),
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": getattr(last_msg, "reasoning_content", "") if last_msg else "",
            "answer": getattr(last_msg, "content", "") or getattr(last_msg, "reasoning_content","") if last_msg else "",
            "messages": messages,
            "reason": reason,
        }

    def benchmark(
            self,
            num_attempts: int,
            save_dir: str,
            save_freq: int = 10,
            max_samples: int = None,
            free_mode: bool = False
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        completions: List[Dict[str, Any]] = []
        save_result: List[Dict[str, Any]] = []

        scenarios = self.environment.get_scenarios()

        if max_samples is not None:
            scenarios = scenarios[:min(max_samples, len(scenarios))]

        # solve each question
        for idx, scenario in enumerate(scenarios):
            scenario_id = scenario.get("scenario_id")
            start_time = time.time()

            n_success = 0.0
            agent_answers: List[str] = []
            sample_response: Optional[Dict[str, Any]] = {}

            # try each attempt
            for attempt in range(num_attempts):
                self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")

                response = self.run(scenario=scenario, free_mode=free_mode)
                sample_response = response

                if response.get("status") == "solved":
                    agent_answer = extract_answer_all(response.get("answer", "")) or extract_answer_all(response.get("traces", ""))
                    # if 'C' not in agent_answer:
                    #     agent_answer = 'C' + agent_answer
                    ground_truth = scenario.get("answer")
                    n_success += compute_score(agent_answer, ground_truth)
                    agent_answers.append(agent_answer)
                    pink = "\033[95m"
                    reset = "\033[0m"
                    self.logger.info(f"{pink}\n[Scenario: {scenario_id}] Agent's answer is {agent_answer}, ground truth is {ground_truth}{reset}.")

            acc = n_success / float(num_attempts)
            latency = round((time.time() - start_time) / float(num_attempts), 2)

            # save completion if needed
            completions.append(
                {
                    "scenario_id": scenario_id,
                    "free_mode": free_mode,
                    "response": (sample_response).get("answer", ""),
                    "traces": (sample_response).get("traces", ""),
                    # "messages": (sample_response).get("messages", ""), # uncomment if needed
                    "num_iterations": (sample_response).get("num_iterations", 0),
                    "num_tool_calls": (sample_response or {}).get("num_tool_calls", 0),
                    "tool_calls": (sample_response).get("tool_calls", []),
                    "answers": agent_answers,
                    "ground_truth": scenario.get("answer"),
                    "accuracy": acc,
                    "latency": latency,
                }
            )

            save_result.append(
                {
                    "scenario_id": scenario_id,
                    "answers": agent_answers[0],
                }
            )

            if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                df = pd.DataFrame(save_result)
                df.to_csv(os.path.join(save_dir, f"result.csv"), index=False)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agents benchmarking")
    parser.add_argument("--server_url", type=str,  default="http://localhost:7860")
    parser.add_argument("--model_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_name", type=str, default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=130)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.log")
    parser.add_argument("--free_mode", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = init_logger(log_file=args.log_file)

    Environment = Environment(server_url=args.server_url, verbose=args.verbose, logger=logger)

    runner = AgentsRunner(
        environment=Environment,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        logger=logger
    )

    runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=args.free_mode,
    )