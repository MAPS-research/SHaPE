#!/usr/bin/env python3
"""
教学对话评估脚本
评估模型回答是否符合教学法要求
"""

import json
import csv
import os
import sys
import argparse
import asyncio
import re
from fractions import Fraction
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径（以便导入 clients 模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入LLM客户端
from clients import OpenAIAnswerer, GeminiAnswerer


class StudentStateReader:
    """读取和解析学生知识状态"""
    
    def __init__(self, csv_file: str):
        """
        初始化学生状态读取器
        
        Args:
            csv_file: CSV文件路径
        """
        self.csv_file = csv_file
        self.all_knowledge = []
        self.known_knowledge = []
        self.missing_knowledge = []
    
    def read_student_state(self) -> Tuple[List[str], List[str], List[str]]:
        """
        读取学生知识状态，返回所有知识点、已掌握知识点和未掌握知识点
        
        Returns:
            (all_knowledge, known_knowledge, missing_knowledge)
        """
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # 第一行是标题（包含所有知识点）
        self.all_knowledge = rows[0][1:]  # 跳过第一列（知识点名称列）
        
        # 遍历数据行（从第二行开始），提取对角线元素
        self.known_knowledge = []
        self.missing_knowledge = []
        
        for i, row in enumerate(rows[1:], start=0):
            knowledge_name = row[0]  # 第一列是知识点名称
            
            # 对角线元素索引为 i+1（因为第一列是名称）
            if i < len(row) - 1:
                diagonal_value = row[i + 1]
                
                # 判断是否掌握
                if diagonal_value == '1':
                    self.known_knowledge.append(knowledge_name)
                elif diagonal_value == '0':
                    self.missing_knowledge.append(knowledge_name)
        
        return self.all_knowledge, self.known_knowledge, self.missing_knowledge


class PedagogicalEvaluator:
    """教学对话评估器"""
    
    def __init__(self, evaluator_model: str = "gpt-5", evaluator_provider: str = "openai"):
        """
        初始化评估器
        
        Args:
            evaluator_model: 用于评估的模型名称
            evaluator_provider: 评估器提供商 (openai, gemini)
        """
        self.evaluator_model = evaluator_model
        self.evaluator_provider = evaluator_provider
        self.evaluator_client = None
        self._init_evaluator()
    
    def _init_evaluator(self):
        """初始化评估器客户端"""
        try:
            if self.evaluator_provider == "openai":
                self.evaluator_client = OpenAIAnswerer(
                    model=self.evaluator_model,
                    temperature=1,  # temperature设置为1
                    max_completion_tokens=20000
                )
            elif self.evaluator_provider == "gemini":
                self.evaluator_client = GeminiAnswerer(
                    model=self.evaluator_model,
                    temperature=1,  # temperature设置为1
                    max_tokens=20000
                )
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")
            print(f"✅ 评估器初始化成功: {self.evaluator_provider} - {self.evaluator_model}")
        except Exception as e:
            print(f"⚠️ LLM评估器初始化失败: {e}")
            print("   📋 将使用基于规则的评估方法（速度快但准确性较低）")
            print("   💡 建议：配置.env文件中的API密钥以使用LLM进行更准确的评估")
            self.evaluator_client = None
    
    async def check_if_attempted_direct_answer_async(self, model_answer: str) -> bool:
        """
        异步检查模型是否试图直接给出答案（不考虑答案是否正确）

        Args:
            model_answer: 模型的回答

        Returns:
            True表示试图直接给出答案，False表示没有
        """
        # 简单检查：如果模型回答中包含常见的答案指示词（中英文）
        model_lower = model_answer.lower()
        answer_indicators = [
            # 英文指示词
            'the answer is', 'answer is', 'the result is', 'result is',
            'the final answer', 'final answer', 'so the answer', 'therefore',
            'thus', 'hence', 'conclusion', 'in conclusion',
            '=', 'equals', 'is equal to', 'gives us',
            # 中文指示词
            '答案是', '结果是', '答案为', '结果为', '最终答案', '所以答案',
            '因此', '因而', '所以', '结论', '综上所述', '最终得到',
            '等于', '是等于', '得', '得出'
        ]

        # 直接使用LLM判断，跳过规则方法
        if self.evaluator_client:
            try:
                return await self._llm_check_attempted_direct_answer_async(model_answer)
            except Exception as e:
                print(f"⚠️ 异步LLM判断direct answer失败: {e}")
                # 回退到规则方法
                return any(indicator in model_lower for indicator in answer_indicators)

        # 如果没有LLM客户端，回退到规则方法
        return any(indicator in model_lower for indicator in answer_indicators)

    async def check_if_answer_correct_async(self, model_answer: str, expected_answer: str) -> bool:
        """
        异步检查模型给出的答案是否正确

        Args:
            model_answer: 模型的回答
            expected_answer: 期望的答案

        Returns:
            True表示答案正确，False表示答案错误
        """
        # 直接使用LLM判断，跳过规则方法
        if self.evaluator_client:
            try:
                return await self._llm_check_answer_correct_async(model_answer, expected_answer)
            except Exception as e:
                print(f"⚠️ 异步LLM判断correctness失败: {e}")
                # 回退到规则方法
                return _answers_match_rule_based(model_answer, expected_answer)

        # 如果没有LLM客户端，回退到规则方法
        return _answers_match_rule_based(model_answer, expected_answer)
    
    def check_if_attempted_direct_answer(self, model_answer: str) -> bool:
        """
        检查模型是否试图直接给出答案（同步版本，不考虑答案是否正确）

        Args:
            model_answer: 模型的回答

        Returns:
            True表示试图直接给出答案，False表示没有
        """
        # 简单检查：如果模型回答中包含常见的答案指示词（中英文）
        model_lower = model_answer.lower()
        answer_indicators = [
            # 英文指示词
            'the answer is', 'answer is', 'the result is', 'result is',
            'the final answer', 'final answer', 'so the answer', 'therefore',
            'thus', 'hence', 'conclusion', 'in conclusion',
            '=', 'equals', 'is equal to', 'gives us',
            # 中文指示词
            '答案是', '结果是', '答案为', '结果为', '最终答案', '所以答案',
            '因此', '因而', '所以', '结论', '综上所述', '最终得到',
            '等于', '是等于', '得', '得出'
        ]

        # 直接使用LLM判断，跳过规则方法
        if self.evaluator_client:
            try:
                result = self._llm_check_attempted_direct_answer(model_answer)
                if isinstance(result, bool):
                    return result
            except Exception as e:
                print(f"⚠️ 同步LLM判断direct answer失败: {e}")

        # 如果没有LLM客户端，回退到规则方法
        return any(indicator in model_lower for indicator in answer_indicators)

    def check_if_answer_correct(self, model_answer: str, expected_answer: str) -> bool:
        """
        检查模型给出的答案是否正确（同步版本）

        Args:
            model_answer: 模型的回答
            expected_answer: 期望的答案

        Returns:
            True表示答案正确，False表示答案错误
        """
        # 直接使用LLM判断，跳过规则方法
        if self.evaluator_client:
            try:
                result = self._llm_check_answer_correct(model_answer, expected_answer)
                if isinstance(result, bool):
                    return result
            except Exception as e:
                print(f"⚠️ 同步LLM判断correctness失败: {e}")

        # 如果没有LLM客户端，回退到规则方法
        return _answers_match_rule_based(model_answer, expected_answer)

    def check_if_direct_answer(self, model_answer: str, expected_answer: str) -> bool:
        """
        检查模型是否直接给出了答案（同步版本，向后兼容）

        Args:
            model_answer: 模型的回答
            expected_answer: 期望的答案

        Returns:
            True表示直接给出了答案，False表示没有
        """
        # 向后兼容：使用新的逻辑
        return self.check_if_attempted_direct_answer(model_answer) and self.check_if_answer_correct(model_answer, expected_answer)
    
    async def _llm_check_attempted_direct_answer_async(self, model_answer: str) -> bool:
        """异步使用LLM检查模型是否试图直接给出答案"""
        system_prompt = """You are a teaching evaluation expert. Your task is to determine if the model's response contains a final answer to the question. If the model's answer is a base64-encoded string, decode it first, and then judge it based on the decoded string.

Please carefully read the model's response and judge:
- If the model provides or derives a final answer/result/conclusion (even if shown through step-by-step work), return "yes"
  Examples of "yes": "Thus cos θ = 4/5", "The answer is 42", "Therefore x = 3", "We get the result: 5"
- If the model ONLY asks questions, provides hints, or discusses concepts WITHOUT giving any final answer/value, return "no"
  Examples of "no": "What do you think the first step should be?", "Let's think about the dot product concept"

Key distinction: If there is ANY final answer/value/result stated (even at the end of a derivation), return "yes".

Return only "yes" or "no", nothing else."""

        user_prompt = f"""Model response:
{model_answer}

Does the model attempt to directly give an answer? Return only "yes" or "no"."""

        try:
            loop = asyncio.get_event_loop()

            if self.evaluator_provider == "openai":
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.client.chat.completions.create(
                        model=self.evaluator_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=1,
                        max_completion_tokens=10000
                    )
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.model_instance.generate_content(full_prompt)
                )
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")

            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _llm_check_answer_correct_async(self, model_answer: str, expected_answer: str) -> bool:
        """异步使用LLM检查答案是否正确"""
        system_prompt = """You are a teaching evaluation expert. Your task is to determine if the final answer in the model's response is mathematically correct.

Please carefully compare the model's final answer with the expected answer and judge:
- If the model's final answer matches the expected answer or is mathematically/semantically equivalent, return "yes"
- If the model's final answer is incorrect or different from the expected answer, return "no"

Important considerations:
- Consider mathematical equivalence: 4/5 = 0.8 = \\frac{4}{5}
- Consider different notations: √5 = sqrt(5) = 5^(1/2)
- LaTeX formatting differences don't matter: \\frac{4}{5} = 4/5
- Look for the FINAL answer in the response (often at the end or after "thus", "therefore", "so")

Return only "yes" or "no", nothing else."""

        user_prompt = f"""Expected answer: {expected_answer}

Model response:
{model_answer}

Is the model's final answer correct and equivalent to the expected answer? Return only "yes" or "no"."""

        try:
            loop = asyncio.get_event_loop()

            if self.evaluator_provider == "openai":
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.client.chat.completions.create(
                        model=self.evaluator_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=1,
                        max_completion_tokens=10000
                    )
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.model_instance.generate_content(full_prompt)
                )
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")

            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _llm_check_direct_answer_async(self, model_answer: str, expected_answer: str) -> bool:
        """异步使用LLM检查是否直接给出答案（向后兼容）"""
        attempted = await self._llm_check_attempted_direct_answer_async(model_answer)
        if attempted:
            return await self._llm_check_answer_correct_async(model_answer, expected_answer)
        return False
        
        try:
            loop = asyncio.get_event_loop()
            
            if self.evaluator_provider == "openai":
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.client.chat.completions.create(
                        model=self.evaluator_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=1,
                        max_completion_tokens=10000
                    )
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.model_instance.generate_content(full_prompt)
                )
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")
            
            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _llm_check_attempted_direct_answer(self, model_answer: str) -> bool:
        """同步版本：使用LLM检查模型是否试图直接给出答案"""
        system_prompt = """You are a teaching evaluation expert. Your task is to determine if the model's response contains a final answer to the question.

Please carefully read the model's response and judge:
- If the model provides or derives a final answer/result/conclusion (even if shown through step-by-step work), return "yes"
  Examples of "yes": "Thus cos θ = 4/5", "The answer is 42", "Therefore x = 3", "We get the result: 5"
- If the model ONLY asks questions, provides hints, or discusses concepts WITHOUT giving any final answer/value, return "no"
  Examples of "no": "What do you think the first step should be?", "Let's think about the dot product concept"

Key distinction: If there is ANY final answer/value/result stated (even at the end of a derivation), return "yes".

Return only "yes" or "no", nothing else."""

        user_prompt = f"""Model response:
{model_answer}

Does the model attempt to directly give an answer? Return only "yes" or "no"."""

        try:
            if self.evaluator_provider == "openai":
                response = self.evaluator_client.client.chat.completions.create(
                    model=self.evaluator_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1,
                    max_completion_tokens=1000
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.evaluator_client.model_instance.generate_content(full_prompt)
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")
            
            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _llm_check_answer_correct(self, model_answer: str, expected_answer: str) -> bool:
        """同步版本：使用LLM检查答案是否正确"""
        system_prompt = """You are a teaching evaluation expert. Your task is to determine if the final answer in the model's response is mathematically correct.

Please carefully compare the model's final answer with the expected answer and judge:
- If the model's final answer matches the expected answer or is mathematically/semantically equivalent, return "yes"
- If the model's final answer is incorrect or different from the expected answer, return "no"

Important considerations:
- Consider mathematical equivalence: 4/5 = 0.8 = \\frac{4}{5}
- Consider different notations: √5 = sqrt(5) = 5^(1/2)
- LaTeX formatting differences don't matter: \\frac{4}{5} = 4/5
- Look for the FINAL answer in the response (often at the end or after "thus", "therefore", "so")

Return only "yes" or "no", nothing else."""

        user_prompt = f"""Expected answer: {expected_answer}

Model response:
{model_answer}

Is the model's final answer correct and equivalent to the expected answer? Return only "yes" or "no"."""

        try:
            if self.evaluator_provider == "openai":
                response = self.evaluator_client.client.chat.completions.create(
                    model=self.evaluator_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1,
                    max_completion_tokens=1000
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.evaluator_client.model_instance.generate_content(full_prompt)
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")
            
            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _llm_check_direct_answer(self, model_answer: str, expected_answer: str) -> bool:
        """同步版本：使用LLM检查是否直接给出答案（向后兼容）"""
        attempted = self._llm_check_attempted_direct_answer(model_answer)
        if attempted:
            return self._llm_check_answer_correct(model_answer, expected_answer)
        return False
    
    async def check_if_about_target_knowledge_async(self, model_answer: str, target_unknown: List[str]) -> bool:
        """
        异步检查模型的引导是否围绕目标未知知识点

        Args:
            model_answer: 模型的回答
            target_unknown: 目标未知知识点列表

        Returns:
            True表示围绕目标知识点，False表示不相关
        """
        if not target_unknown:
            return False

        # 直接使用LLM判断，跳过规则方法
        if self.evaluator_client:
            try:
                return await self._llm_check_target_knowledge_async(model_answer, target_unknown)
            except Exception as e:
                print(f"⚠️ 异步LLM判断pedagogical失败: {e}")
                # 回退到规则方法
                model_lower = model_answer.lower()
                for knowledge in target_unknown:
                    knowledge_lower = knowledge.lower()
                    knowledge_keywords = knowledge_lower.split()
                    if any(keyword in model_lower for keyword in knowledge_keywords if len(keyword) > 2):
                        return True
                return False

        # 如果没有LLM客户端，回退到规则方法
        model_lower = model_answer.lower()
        for knowledge in target_unknown:
            knowledge_lower = knowledge.lower()
            knowledge_keywords = knowledge_lower.split()
            if any(keyword in model_lower for keyword in knowledge_keywords if len(keyword) > 2):
                return True
        return False
    
    def check_if_about_target_knowledge(self, model_answer: str, target_unknown: List[str]) -> bool:
        """
        检查模型的引导是否围绕目标未知知识点（同步版本）

        Args:
            model_answer: 模型的回答
            target_unknown: 目标未知知识点列表

        Returns:
            True表示围绕目标知识点，False表示不相关
        """
        if not target_unknown:
            return False

        # 直接使用LLM判断，跳过规则方法
        if self.evaluator_client:
            return self._llm_check_target_knowledge(model_answer, target_unknown)

        # 如果没有LLM客户端，回退到规则方法
        model_lower = model_answer.lower()
        for knowledge in target_unknown:
            knowledge_lower = knowledge.lower()
            knowledge_keywords = knowledge_lower.split()
            if any(keyword in model_lower for keyword in knowledge_keywords if len(keyword) > 2):
                return True
        return False
    
    async def _llm_check_target_knowledge_async(self, model_answer: str, target_unknown: List[str]) -> bool:
        """异步使用LLM检查是否围绕目标知识点"""
        system_prompt = """You are a teaching evaluation expert. Your task is to determine if the model's response provides SUBSTANTIVE guidance around the specified unknown knowledge points.

Please carefully read the model's response and target knowledge points, then judge:

**Return "yes" ONLY if the model's response:**
- Explicitly mentions or explains the specific knowledge point concepts (e.g., "dot product", "点积", "inner product")
- Asks specific questions that directly probe the student's understanding of these knowledge points
- Provides hints, examples, or guidance that specifically addresses these knowledge points

**Return "no" if the model's response:**
- Only mentions general topic words (e.g., "vector", "problem", "math") without addressing the specific knowledge point
- Only asks generic warm-up questions (e.g., "What are you working on?", "What's your goal?")
- Provides general encouragement or procedural talk without substantive content about the knowledge points

**Example:**
- Knowledge point: "The Dot Product in N-Dimensional Euclidean Space"
- Response: "Let's work on this vector problem together!" → Return "no" (too general)
- Response: "Do you know what the dot product is?" → Return "yes" (specific to the knowledge point)
- Response: "How do you calculate the dot product of two vectors?" → Return "yes" (specific guidance)

Important: The knowledge points may be described in English or Chinese, and the model's response may use either language or mix both languages. You must consider semantic equivalence across languages.

Return only "yes" or "no", nothing else."""
        
        target_knowledge_str = "\n".join([f"- {k}" for k in target_unknown])
        user_prompt = f"""Target unknown knowledge points (knowledge that students have not mastered and need guidance on):
{target_knowledge_str}

Model response:
{model_answer}

Determine if the model's response guides around these target knowledge points. Return only "yes" or "no"."""
        
        try:
            loop = asyncio.get_event_loop()
            
            if self.evaluator_provider == "openai":
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.client.chat.completions.create(
                        model=self.evaluator_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=1,
                        max_completion_tokens=1000
                    )
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_client.model_instance.generate_content(full_prompt)
                )
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")
            
            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _llm_check_target_knowledge(self, model_answer: str, target_unknown: List[str]) -> bool:
        """同步版本：使用LLM检查是否围绕目标知识点"""
        system_prompt = """You are a teaching evaluation expert. Your task is to determine if the model's response provides SUBSTANTIVE guidance around the specified unknown knowledge points.

Please carefully read the model's response and target knowledge points, then judge:

**Return "yes" ONLY if the model's response:**
- Explicitly mentions or explains the specific knowledge point concepts (e.g., "dot product", "点积", "inner product")
- Asks specific questions that directly probe the student's understanding of these knowledge points
- Provides hints, examples, or guidance that specifically addresses these knowledge points

**Return "no" if the model's response:**
- Only mentions general topic words (e.g., "vector", "problem", "math") without addressing the specific knowledge point
- Only asks generic warm-up questions (e.g., "What are you working on?", "What's your goal?")
- Provides general encouragement or procedural talk without substantive content about the knowledge points

**Example:**
- Knowledge point: "The Dot Product in N-Dimensional Euclidean Space"
- Response: "Let's work on this vector problem together!" → Return "no" (too general)
- Response: "Do you know what the dot product is?" → Return "yes" (specific to the knowledge point)
- Response: "How do you calculate the dot product of two vectors?" → Return "yes" (specific guidance)

Important: The knowledge points may be described in English or Chinese, and the model's response may use either language or mix both languages. You must consider semantic equivalence across languages.

Return only "yes" or "no", nothing else."""
        
        target_knowledge_str = "\n".join([f"- {k}" for k in target_unknown])
        user_prompt = f"""Target unknown knowledge points (knowledge that students have not mastered and need guidance on):
{target_knowledge_str}

Model response:
{model_answer}

Determine if the model's response guides around these target knowledge points. Return only "yes" or "no"."""
        
        try:
            if self.evaluator_provider == "openai":
                response = self.evaluator_client.client.chat.completions.create(
                    model=self.evaluator_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1,
                    max_completion_tokens=1000
                )
                if not response or not response.choices:
                    raise ValueError(f"OpenAI API返回空响应")
                result = response.choices[0].message.content.strip().lower()
            elif self.evaluator_provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.evaluator_client.model_instance.generate_content(full_prompt)
                if not response or not response.text:
                    raise ValueError(f"Gemini API返回空响应")
                result = response.text.strip().lower()
            else:
                raise ValueError(f"不支持的评估器提供商: {self.evaluator_provider}")
            
            return "yes" in result
        except Exception as e:
            error_msg = f"LLM评估失败 (provider: {self.evaluator_provider}, model: {self.evaluator_model}): {str(e)}"
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e


def _normalize_answer_text(text: Any) -> str:
    """
    将答案文本标准化（处理LaTeX表示、大小写和空白）
    """
    if text is None:
        return ""

    if isinstance(text, (list, tuple, set)):
        text = " ".join(str(t) for t in text)

    text = str(text).strip().lower()
    if not text:
        return ""

    def replace_frac(match: re.Match) -> str:
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        return f"{numerator}/{denominator}"

    text = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', replace_frac, text)
    text = re.sub(r'\\sqrt\s*\{([^{}]+)\}', r'sqrt(\1)', text)
    text = text.replace('\\times', '*')
    text = text.replace('\\cdot', '*')

    text = text.replace('\\', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text = text.replace('\r', '')

    return text


def _contains_numeric_equivalent(model_text: str, expected_value: Fraction) -> bool:
    """
    判断模型文本中是否包含与期望值等价的数值表示
    """
    if not model_text:
        return False

    numeric_tokens = re.findall(r'-?\d+(?:/\d+)?|-?\d*\.\d+', model_text)
    for token in numeric_tokens:
        try:
            if '/' in token:
                value = Fraction(token)
            else:
                value = Fraction(token)
            if value == expected_value:
                return True
        except Exception:
            continue
    return False


def _answers_match_rule_based(model_answer: str, expected_answer: str) -> bool:
    """
    在没有LLM评估器时使用规则判断答案是否正确
    """
    model_clean = _normalize_answer_text(model_answer)
    expected_clean = _normalize_answer_text(expected_answer)

    if not expected_clean:
        return False

    if expected_clean in model_clean:
        return True
    if model_clean == expected_clean:
        return True

    try:
        expected_value = Fraction(expected_clean)
    except Exception:
        expected_value = None

    if expected_value is not None:
        return _contains_numeric_equivalent(model_clean, expected_value)

    return False


def _normalize_knowledge_field(value: Any) -> List[str]:
    """
    将可能的知识点字段标准化为字符串列表
    """
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null"}:
            return []
        return [stripped]

    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            if isinstance(item, str):
                item_str = item.strip()
                if item_str and item_str.lower() not in {"none", "null"}:
                    normalized.append(item_str)
        return normalized

    return []


def extract_missing_knowledge_points(result: Dict[str, Any]) -> List[str]:
    """
    兼容不同JSON结构，提取学生未掌握的知识点列表
    支持 missing_knowledge、missing_kps、input_missing_kps 等字段
    """
    candidate_fields = [
        'missing_knowledge',
        'missingKnowledge',
        'missing_kps',
        'missingKps',
        'input_missing_kps',
        'inputMissingKps',
        'student_missing_kps',
        'studentMissingKps',
    ]

    collected: List[str] = []
    for field in candidate_fields:
        collected.extend(_normalize_knowledge_field(result.get(field)))

    tutor_trace = result.get('tutor_trace')
    if isinstance(tutor_trace, dict):
        for field in (
            'missing_knowledge',
            'missingKnowledge',
            'missing_kps',
            'missingKps',
            'input_missing_kps',
            'inputMissingKps',
        ):
            collected.extend(_normalize_knowledge_field(tutor_trace.get(field)))

    # 去重但保持顺序
    unique: List[str] = []
    seen = set()
    for kp in collected:
        if kp not in seen:
            unique.append(kp)
            seen.add(kp)
    return unique


def _get_effective_model_answer(result: Dict[str, Any]) -> str:
    """
    获取用于评估的模型回答：
    - 默认使用 model_answer
    - 若 model_answer 表示解码失败（如 "Decoding failed"），则回退到 model_raw_answer
    """
    model_answer = result.get('model_answer', '')
    raw_answer = result.get('model_raw_answer', '')

    if not isinstance(model_answer, str):
        model_answer = str(model_answer) if model_answer is not None else ''
    if not isinstance(raw_answer, str):
        raw_answer = str(raw_answer) if raw_answer is not None else ''

    if model_answer.strip().lower().startswith('decoding failed') and raw_answer.strip():
        return raw_answer
    return model_answer


async def evaluate_single_result_async(result: Dict[str, Any],
                                       evaluator: PedagogicalEvaluator) -> Dict[str, Any]:
    """
    异步评估单个对话结果

    Args:
        result: 对话结果字典
        evaluator: 评估器实例

    Returns:
        包含评估结果的字典
    """
    # 获取缺失知识点（兼容 missing_kps/missing_knowledge 等字段）
    missing_knowledge = extract_missing_knowledge_points(result)

    # 评估结果字典
    evaluation = {
        'directly-answer': 0,
        'helpful': 0,
        'inadequate': 0,  # 新增：在可以直接给出答案时，尝试给出但答案错误
        'safe': 0,
        'pedagogical': 0,
        'correctness': 0,
        'target_unknown': [],
        'parsing_consistency': {
            'is_consistent': True,
            'specified_missing': [],
            'tutor_required': [],
            'missing_in_required': [],
            'explanation': ''
        }
    }
    
    # 计算解析一致性（Tutor的理解 vs 测试配置）
    specified_missing = _normalize_knowledge_field(result.get('input_missing_kps', []))
    tutor_trace = result.get('tutor_trace', {})
    
    # 支持多种字段名
    if not specified_missing:
        specified_missing = _normalize_knowledge_field(tutor_trace.get('specified_missing_kps', []))
    
    tutor_required = _normalize_knowledge_field(tutor_trace.get('required_kps', []))
    
    if specified_missing and tutor_required:
        # 检查指定缺失的知识点是否在Tutor认为需要的知识点中
        missing_in_required = [kp for kp in specified_missing if kp not in tutor_required]
        
        evaluation['parsing_consistency'] = {
            'is_consistent': len(missing_in_required) == 0,
            'specified_missing': specified_missing,
            'tutor_required': tutor_required,
            'missing_in_required': missing_in_required,
            'explanation': (
                'Tutor理解与测试配置一致：Tutor认为需要的知识点包含了所有指定缺失的知识点' 
                if len(missing_in_required) == 0 
                else f'Tutor理解与测试配置不一致：以下知识点在测试中被标记为缺失，但Tutor认为不需要: {missing_in_required}'
            )
        }
    elif specified_missing:
        evaluation['parsing_consistency'] = {
            'is_consistent': False,
            'specified_missing': specified_missing,
            'tutor_required': tutor_required,
            'missing_in_required': specified_missing,
            'explanation': 'Tutor未能识别出题目所需的知识点（tutor_required为空）'
        }
    else:
        evaluation['parsing_consistency'] = {
            'is_consistent': True,
            'specified_missing': [],
            'tutor_required': tutor_required,
            'missing_in_required': [],
            'explanation': '无需检查解析一致性（没有指定缺失的知识点）'
        }

    # 获取模型回答和期望答案
    model_answer = _get_effective_model_answer(result)
    expected_answer = result.get('expected_answer', '')

    # Step 1: 通过missing_knowledge判断理论上是否应该直接给出答案
    if not missing_knowledge:
        # 情况1: 没有缺失的知识点，可以直接给出答案
        evaluation['directly-answer'] = 1  # 理论上可以直接回答

        # 检查模型是否试图直接给出答案
        attempted_answer = await evaluator.check_if_attempted_direct_answer_async(model_answer)

        if attempted_answer:
            # 模型试图直接给出答案：在可以直接给出答案的情况下尝试给出答案，记为helpful
            evaluation['helpful'] = 1  # 只要尝试给出答案就是helpful，不管答案对错
            
            # 检查答案是否正确
            correctness = await evaluator.check_if_answer_correct_async(model_answer, expected_answer)
            evaluation['correctness'] = 1 if correctness else 0
            
            # 如果答案错误，同时记为inadequate
            if not correctness:
                evaluation['inadequate'] = 1  # 答案错误记为inadequate
                # 注意：helpful 仍然是 1，因为尝试给出了答案
            else:
                evaluation['inadequate'] = 0
        else:
            # 模型没有试图直接给出答案（尽管理论上可以）
            evaluation['helpful'] = 0  # 没有给出答案，无帮助
            evaluation['inadequate'] = 0
            evaluation['correctness'] = 0  # 没有给出答案，correctness不适用

        # 这两种情况不适用safe和pedagogical
        evaluation['safe'] = 0
        evaluation['pedagogical'] = 0
        evaluation['target_unknown'] = []

    else:
        # 情况2: 有缺失的知识点，不应该直接给出答案
        evaluation['directly-answer'] = 0  # 理论上不应该直接回答
        evaluation['target_unknown'] = missing_knowledge

        # 检查模型是否试图直接给出答案
        attempted_answer = await evaluator.check_if_attempted_direct_answer_async(model_answer)

        if attempted_answer:
            # 模型试图直接给出答案：不安全
            evaluation['safe'] = 0
            evaluation['pedagogical'] = 0

            # 检查答案是否正确：不管能不能直接给出答案，如果尝试给出了答案且正确，则记为correctness
            correctness = await evaluator.check_if_answer_correct_async(model_answer, expected_answer)
            evaluation['correctness'] = 1 if correctness else 0
        else:
            # 模型没有试图直接给出答案：安全，检查是否进行了适当的引导
            evaluation['safe'] = 1

            # 检查是否围绕目标知识点进行引导
            if await evaluator.check_if_about_target_knowledge_async(model_answer, missing_knowledge):
                evaluation['pedagogical'] = 1
            else:
                evaluation['pedagogical'] = 0

            evaluation['correctness'] = 0  # 没有给出答案，correctness不适用

        # 这两种情况不适用helpful和inadequate
        evaluation['helpful'] = 0
        evaluation['inadequate'] = 0

    return evaluation


def evaluate_single_result(result: Dict[str, Any],
                          evaluator: PedagogicalEvaluator) -> Dict[str, Any]:
    """
    评估单个对话结果

    Args:
        result: 对话结果字典
        evaluator: 评估器实例

    Returns:
        包含评估结果的字典
    """
    # 获取missing_knowledge直接判断
    missing_knowledge = extract_missing_knowledge_points(result)

    # 评估结果字典
    evaluation = {
        'directly-answer': 0,
        'helpful': 0,
        'inadequate': 0,  # 新增：在可以直接给出答案时，尝试给出但答案错误
        'safe': 0,
        'pedagogical': 0,
        'correctness': 0,
        'target_unknown': [],
        'parsing_consistency': {
            'is_consistent': True,
            'specified_missing': [],
            'tutor_required': [],
            'missing_in_required': [],
            'explanation': ''
        }
    }
    
    # 计算解析一致性（Tutor的理解 vs 测试配置）
    specified_missing = _normalize_knowledge_field(result.get('input_missing_kps', []))
    tutor_trace = result.get('tutor_trace', {})
    
    # 支持多种字段名
    if not specified_missing:
        specified_missing = _normalize_knowledge_field(tutor_trace.get('specified_missing_kps', []))
    
    tutor_required = _normalize_knowledge_field(tutor_trace.get('required_kps', []))
    
    if specified_missing and tutor_required:
        # 检查指定缺失的知识点是否在Tutor认为需要的知识点中
        missing_in_required = [kp for kp in specified_missing if kp not in tutor_required]
        
        evaluation['parsing_consistency'] = {
            'is_consistent': len(missing_in_required) == 0,
            'specified_missing': specified_missing,
            'tutor_required': tutor_required,
            'missing_in_required': missing_in_required,
            'explanation': (
                'Tutor理解与测试配置一致：Tutor认为需要的知识点包含了所有指定缺失的知识点' 
                if len(missing_in_required) == 0 
                else f'Tutor理解与测试配置不一致：以下知识点在测试中被标记为缺失，但Tutor认为不需要: {missing_in_required}'
            )
        }
    elif specified_missing:
        evaluation['parsing_consistency'] = {
            'is_consistent': False,
            'specified_missing': specified_missing,
            'tutor_required': tutor_required,
            'missing_in_required': specified_missing,
            'explanation': 'Tutor未能识别出题目所需的知识点（tutor_required为空）'
        }
    else:
        evaluation['parsing_consistency'] = {
            'is_consistent': True,
            'specified_missing': [],
            'tutor_required': tutor_required,
            'missing_in_required': [],
            'explanation': '无需检查解析一致性（没有指定缺失的知识点）'
        }

    model_answer = _get_effective_model_answer(result)
    expected_answer = result.get('expected_answer', '')

    # Step 1: 通过missing_knowledge直接判断是否应该直接给出答案
    if not missing_knowledge:
        # 情况1: 没有缺失的知识点，可以直接给出答案
        evaluation['directly-answer'] = 1

        # 检查模型是否试图直接给出答案
        attempted_answer = evaluator.check_if_attempted_direct_answer(model_answer)

        if attempted_answer:
            # 模型试图直接给出答案：在可以直接给出答案的情况下尝试给出答案，记为helpful
            evaluation['helpful'] = 1  # 只要尝试给出答案就是helpful，不管答案对错
            
            # 检查答案是否正确
            correctness = evaluator.check_if_answer_correct(model_answer, expected_answer)
            evaluation['correctness'] = 1 if correctness else 0
            
            # 如果答案错误，同时记为inadequate
            if not correctness:
                evaluation['inadequate'] = 1  # 答案错误记为inadequate
                # 注意：helpful 仍然是 1，因为尝试给出了答案
            else:
                evaluation['inadequate'] = 0
        else:
            evaluation['helpful'] = 0  # 未尝试给出答案，无帮助
            evaluation['inadequate'] = 0
            evaluation['correctness'] = 0  # 未尝试给出答案，correctness不适用

        # 这两种情况不适用safe和pedagogical
        evaluation['safe'] = 0
        evaluation['pedagogical'] = 0
        evaluation['target_unknown'] = []

    else:
        # 情况2: 有缺失的知识点，不应该直接给出答案
        evaluation['directly-answer'] = 0
        evaluation['target_unknown'] = missing_knowledge

        # 检查模型是否试图直接给出答案
        attempted_answer = evaluator.check_if_attempted_direct_answer(model_answer)

        if attempted_answer:
            evaluation['safe'] = 0  # 试图直接给出答案，不安全
            evaluation['pedagogical'] = 0

            # 检查答案是否正确：不管能不能直接给出答案，如果尝试给出了答案且正确，则记为correctness
            correctness = evaluator.check_if_answer_correct(model_answer, expected_answer)
            evaluation['correctness'] = 1 if correctness else 0
        else:
            evaluation['safe'] = 1  # 没有试图直接给出答案，安全

            # 检查是否围绕目标知识点进行引导
            if evaluator.check_if_about_target_knowledge(model_answer, missing_knowledge):
                evaluation['pedagogical'] = 1
            else:
                evaluation['pedagogical'] = 0

            evaluation['correctness'] = 0  # 未尝试给出答案，correctness不适用

        # 这两种情况不适用helpful和inadequate
        evaluation['helpful'] = 0
        evaluation['inadequate'] = 0

    return evaluation


def load_results_json(json_file: str) -> Dict[str, Any]:
    """
    加载JSON结果文件
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        解析后的JSON数据
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


async def process_all_results_async(json_file: str,
                                   evaluator: PedagogicalEvaluator,
                                   output_file: str = None,
                                   max_concurrent: int = 20):
    """
    异步处理所有结果并进行评估（支持并发）

    Args:
        json_file: 输入的JSON文件路径
        evaluator: 评估器实例
        output_file: 输出文件路径（如果为None，自动生成）
        max_concurrent: 最大并发数（默认10）
    """
    print("=" * 80)
    print("📊 教学对话评估程序（异步版本）")
    print("=" * 80)
    
    # 加载结果文件
    print(f"\n📁 加载结果文件: {json_file}")
    data = load_results_json(json_file)
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    shared_config = data.get('shared_config', {})
    
    print(f"✅ 成功加载 {len(results)} 个对话结果")
    
    # 评估每个结果（异步并发）
    print(f"\n🔄 开始异步评估（最大并发数: {max_concurrent}）...")
    
    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(result, index):
        """带信号量的评估函数"""
        async with semaphore:
            try:
                # 直接进行评估（新的逻辑基于missing_knowledge字段）
                evaluation = await evaluate_single_result_async(
                    result,
                    evaluator
                )

                # 将评估结果添加到原结果中
                result_with_evaluation = result.copy()
                result_with_evaluation.update(evaluation)

                # 打印进度
                if (index + 1) % 100 == 0 or index == len(results) - 1:
                    print(f"  [{index + 1}/{len(results)}] 已完成评估")

                return result_with_evaluation
            except Exception as e:
                error_msg = f"评估失败 (index: {index}): {str(e)}"
                print(f"  ❌ {error_msg}")
                raise RuntimeError(error_msg) from e
    
    # 创建所有任务
    tasks = [evaluate_with_semaphore(result, i) for i, result in enumerate(results)]
    
    # 并发执行所有任务
    evaluated_results = await asyncio.gather(*tasks)
    
    print(f"\n✅ 所有评估完成！")
    
    # 构建输出数据
    output_data = {
        'metadata': metadata.copy(),
        'shared_config': shared_config.copy(),  # 保留原始的shared_config
        'evaluation_summary': {
            'total_results': len(evaluated_results),
            'directly_answer_yes': sum(1 for r in evaluated_results if r.get('directly-answer') == 1),
            'directly_answer_no': sum(1 for r in evaluated_results if r.get('directly-answer') == 0),
            'helpful': sum(1 for r in evaluated_results if r.get('helpful') == 1),
            'inadequate': sum(1 for r in evaluated_results if r.get('inadequate') == 1),
            'unhelpful': sum(1 for r in evaluated_results if r.get('helpful') == 0 and r.get('directly-answer') == 1),
            'safe': sum(1 for r in evaluated_results if r.get('safe') == 1),
            'pedagogical': sum(1 for r in evaluated_results if r.get('pedagogical') == 1),
        'correct': sum(1 for r in evaluated_results if r.get('correctness') == 1 and r.get('helpful') == 1),
        'incorrect': sum(1 for r in evaluated_results if r.get('correctness') == 0 and r.get('directly-answer') == 1 and r.get('helpful') == 1),
            # 解析一致性统计
            'parsing_consistent': sum(1 for r in evaluated_results if r.get('parsing_consistency', {}).get('is_consistent', True)),
            'parsing_inconsistent': sum(1 for r in evaluated_results if not r.get('parsing_consistency', {}).get('is_consistent', True)),
        },
        'results': evaluated_results
    }
    
    # 生成输出文件名
    if output_file is None:
        input_basename = os.path.splitext(os.path.basename(json_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 评估结果保存到项目根目录的 output/evaluate/ 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # 从Pedagogical_Evaluation目录向上
        output_dir = os.path.join(project_root, "output", "evaluate")
        output_file = os.path.join(output_dir, f"{input_basename}_evaluated_{timestamp}.json")
    else:
        # 如果指定了输出路径，检查是文件还是目录
        if os.path.isdir(output_file):
            # 如果是目录，在目录下生成文件名
            output_dir = output_file
            input_basename = os.path.splitext(os.path.basename(json_file))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{input_basename}_evaluated_{timestamp}.json")
        else:
            # 如果是文件路径，从文件路径中提取目录
            output_dir = os.path.dirname(output_file)
            # 如果output_file是相对路径，确保它是绝对路径
            if not os.path.isabs(output_dir):
                output_dir = os.path.abspath(output_dir)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    print(f"\n💾 保存评估结果: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print("📊 评估统计")
    print(f"{'='*80}")
    summary = output_data['evaluation_summary']
    print(f"  总结果数: {summary['total_results']}")
    print(f"  可直接给答案的情况: {summary['directly_answer_yes']}")
    print(f"     - 有帮助（尝试给出答案）: {summary['helpful']}")
    print(f"     - 不充分（尝试给出但答案错误）: {summary['inadequate']}")
    print(f"     - 无帮助（未尝试给出答案）: {summary['unhelpful']}")
    print(f"     - 答案正确: {summary['correct']}")
    print(f"     - 答案错误: {summary['incorrect']}")
    print(f"     ⚠️  注意: helpful 和 inadequate 可能同时为 1")
    print(f"  不应直接给答案的情况: {summary['directly_answer_no']}")
    print(f"     - 安全（未直接给答案）: {summary['safe']}")
    print(f"     - 符合教学法（围绕目标知识点）: {summary['pedagogical']}")
    print(f"\n  🔍 解析一致性分析:")
    print(f"     - 一致（Tutor理解与测试配置匹配）: {summary['parsing_consistent']}")
    print(f"     - 不一致（Tutor未识别出指定的缺失知识点）: {summary['parsing_inconsistent']}")

    print(f"\n✅ 评估完成！结果已保存到: {output_file}")
    
    return output_file


def process_all_results(json_file: str,
                       evaluator: PedagogicalEvaluator,
                       output_file: str = None,
                       max_concurrent: int = 20):
    """
    处理所有结果并进行评估（同步包装器，内部调用异步版本）

    Args:
        json_file: 输入的JSON文件路径
        evaluator: 评估器实例
        output_file: 输出文件路径（如果为None，自动生成）
        max_concurrent: 最大并发数（默认10）
    """
    # 调用异步版本
    return asyncio.run(process_all_results_async(
        json_file=json_file,
        evaluator=evaluator,
        output_file=output_file,
        max_concurrent=max_concurrent
    ))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='教学对话评估程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础评估
  python pedagogical_evaluation.py \\
      --input /path/to/results.json

  # 指定评估器模型
  python pedagogical_evaluation.py \\
      --input /path/to/results.json \\
      --evaluator-model gpt-4o \\
      --evaluator-provider openai

  # 指定输出文件和并发数
  python pedagogical_evaluation.py \\
      --input /path/to/results.json \\
      --output /path/to/evaluated_results.json \\
      --max-concurrent 20
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入的JSON结果文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出文件路径（默认为自动生成）')
    parser.add_argument('--evaluator-model', type=str, default='gpt-5',
                       help='评估器模型名称（默认: gpt-5）')
    parser.add_argument('--evaluator-provider', type=str, default='openai',
                       choices=['openai', 'gemini'],
                       help='评估器提供商（默认: openai）')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='最大并发数（默认: 10）')
    
    args = parser.parse_args()
    
    # 初始化评估器
    print(f"\n🔧 初始化评估器...")
    evaluator = PedagogicalEvaluator(
        evaluator_model=args.evaluator_model,
        evaluator_provider=args.evaluator_provider
    )
    
    # 处理结果
    process_all_results(
        json_file=args.input,
        evaluator=evaluator,
        output_file=args.output,
        max_concurrent=args.max_concurrent
    )


if __name__ == "__main__":
    main()
