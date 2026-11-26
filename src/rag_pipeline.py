
from .retriever import Retriever
from .llm import load_llm
import torch


class RAGPipeline:

    def __init__(self):
        print("Initializing RAG Pipeline...")

        self.retriever = Retriever()
        self.tokenizer, self.model = load_llm()

        print("RAG Pipeline initialized successfully!")

    def prepare_context(self, question, k=5, sub_questions=None):

        if sub_questions:
            results = self.retrieve_multiple_queries(
                sub_questions,
                k=k
            )
        else:
            results = self.retriever.retrieve(question, k=k)

        context = ""

        for result in results:

            chunk = result["chunk"]

            context += (
                f"\nSource: {chunk['source']}\n"
                f"Page: {chunk['page']}\n"
                f"Content: {chunk['text']}\n"
            )

        return context, results
    
    def retrieve_multiple_queries(self, questions, k=3):

        all_results = []
        seen_chunks = set()

        for question in questions:

            results = self.retriever.retrieve(question, k=k)

            for result in results:

                chunk = result["chunk"]
                chunk_id = chunk["chunk_id"]

                if chunk_id not in seen_chunks:

                    all_results.append(result)
                    seen_chunks.add(chunk_id)

        return all_results
    
    def generate_answer(self, question, k=3, sub_questions=None):

        context, results = self.prepare_context(
            question,
            k=k,
            sub_questions=sub_questions
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are UniGuide AI, an assistant that provides accurate "
                    "information about scholarships and study opportunities. "

                    "Answer using only the supplied context. "

                    "Carefully inspect all relevant information in the context "
                    "before concluding that an answer is unavailable. "

                    "If the context contains information that partially or "
                    "indirectly answers the question, explain what it establishes "
                    "and clearly identify what it does not establish. "

                    "Distinguish general eligibility and scholarship rules "
                    "from nationality-specific rules. Never assume that a "
                    "general rule guarantees eligibility for a particular student. "

                    "Do not invent information, amounts, dates, or conditions. "

                    "If the requested information is absent, clearly say "
                    "that it is not available in the provided documents. "

                    "Be clear, concise, and professional. "
                    "Mention the source and page when relevant."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        # Send inputs to the same device as the model
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        print("\nINPUT DEBUG")
        print("-----------")
        print("Question:", question)
        print("Input tokens:", inputs["input_ids"].shape[1])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode only newly generated tokens
        answer_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True
        )

        # Collect retrieved source references
        sources = []
        seen_sources = set()

        for result in results:

            chunk = result["chunk"]

            source = chunk["source"]
            page = chunk["page"]

            reference = f"{source} — Page {page}"

            if reference not in seen_sources:
                sources.append(reference)
                seen_sources.add(reference)

        # Add references to the answer
        if sources:
            answer += "\n\n**Retrieved references:**\n"

            for source in sources:
                answer += f"- {source}\n"

        return answer
