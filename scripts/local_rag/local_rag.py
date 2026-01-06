#!/usr/bin/env python3

"""
Fully local RAG system - no API keys, no internet needed
Uses: sentence-transformers (embeddings) + Ollama (LLM)
Fixed: Better text normalization for PDFs with uneven spacing
Updated: Works from any directory using absolute paths
"""

from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import subprocess
import sys
import re
import os


class LocalRAG:
    def __init__(self, collection_name="my_pdfs", pdf_folder=None):
        print("🔧 Initializing Local RAG...")

        # Use a consistent data directory in user's home
        self.data_dir = Path.home() / ".local" / "share" / "local_rag"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load local embedding model (downloads on first run)
        print("📥 Loading embedding model (first time may take a minute)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")

        # Initialize local vector database in data directory
        db_path = str(self.data_dir / "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # Try to get existing collection or create new one
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            count = self.collection.count()
            print(f"✅ Loaded existing collection: {collection_name} ({count} chunks)")

            # Try to restore PDF folder from collection metadata
            if not pdf_folder:
                config_file = self.data_dir / "chroma_db" / ".pdf_folder"
                if config_file.exists():
                    stored_folder = config_file.read_text().strip()
                    self.pdf_folder = Path(stored_folder).expanduser().resolve()
                    print(f"📁 Restored PDF folder: {self.pdf_folder}")
                else:
                    self.pdf_folder = None
                    print(
                        f"⚠️  PDF folder not set. Use 'setfolder' command or re-index."
                    )
            else:
                self.pdf_folder = Path(pdf_folder).expanduser().resolve()
                # Save folder location for future sessions
                config_file = self.data_dir / "chroma_db" / ".pdf_folder"
                config_file.parent.mkdir(parents=True, exist_ok=True)
                config_file.write_text(str(self.pdf_folder))

        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            print(f"✅ Created new collection: {collection_name}")
            self.pdf_folder = (
                Path(pdf_folder).expanduser().resolve() if pdf_folder else None
            )

    def list_indexed_pdfs(self):
        """Show all PDFs that have been indexed"""
        results = self.collection.get()

        if not results["metadatas"]:
            print("❌ No documents indexed")
            return

        # Get unique files
        files = {}
        for metadata in results["metadatas"]:
            filename = metadata["file"]
            page = metadata["page"]
            if filename not in files:
                files[filename] = {"pages": set(), "chunks": 0}
            files[filename]["pages"].add(page)
            files[filename]["chunks"] += 1

        print(
            f"\n📚 Indexed PDFs ({len(files)} files, {len(results['metadatas'])} total chunks):\n"
        )
        for filename in sorted(files.keys()):
            info = files[filename]
            pages_str = f"{min(info['pages'])}-{max(info['pages'])}"
            print(f"   ✓ {filename}")
            print(f"      Pages: {pages_str} ({len(info['pages'])} pages)")
            print(f"      Chunks: {info['chunks']}")

        return files

    def _normalize_text(self, text):
        """Aggressive text normalization for PDFs with spacing issues"""
        if not text:
            return ""

        # Replace multiple whitespace (spaces, tabs, newlines) with single space
        text = re.sub(r"\s+", " ", text)

        # Remove zero-width spaces and other invisible characters
        text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        # Normalize dashes
        text = text.replace("–", "-").replace("—", "-")

        # Remove excessive punctuation spacing
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text.strip()

    def index_pdfs(
        self,
        folder_path,
        chunk_size=500,
        chunk_overlap=50,
        batch_size=500,
        num_workers=4,
    ):
        """Index all PDFs in a folder with parallel processing and large batches"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        folder = Path(folder_path).expanduser().resolve()
        self.pdf_folder = folder

        # Save folder location for future sessions
        config_file = self.data_dir / "chroma_db" / ".pdf_folder"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(str(folder))

        pdf_files = sorted(list(folder.glob("*.pdf")))

        if not pdf_files:
            print(f"❌ No PDFs found in {folder_path}")
            return

        print(f"\n📚 Indexing {len(pdf_files)} PDFs...")
        print(f"   Chunk size: {chunk_size} chars, overlap: {chunk_overlap}")
        print(f"   Batch size: {batch_size} chunks")
        print(f"   Workers: {num_workers} parallel threads")
        print("")

        start_time = time.time()
        total_chunks = 0

        # Extract text from all PDFs in parallel
        def extract_pdf_text(pdf_file):
            """Extract and chunk text from a single PDF"""
            chunks_data = []
            try:
                with open(pdf_file, "rb") as file:
                    reader = PdfReader(file)

                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if not text or len(text.strip()) < 50:
                            continue

                        # Aggressive normalization for uneven spacing
                        text = self._normalize_text(text)

                        if len(text.strip()) < 50:
                            continue

                        # Create overlapping chunks
                        for i in range(0, len(text), chunk_size - chunk_overlap):
                            chunk = text[i : i + chunk_size]
                            if len(chunk.strip()) < 50:
                                continue

                            chunks_data.append(
                                {
                                    "text": chunk,
                                    "file": pdf_file.name,
                                    "page": page_num + 1,
                                }
                            )

                return pdf_file.name, chunks_data, None
            except Exception as e:
                return pdf_file.name, [], str(e)

        # Process PDFs in parallel
        print("🔄 Extracting text from PDFs...")
        all_chunks = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_pdf_text, pdf) for pdf in pdf_files]

            for future in as_completed(futures):
                filename, chunks_data, error = future.result()
                if error:
                    print(f"✗ {filename}: {error}")
                else:
                    all_chunks.extend(chunks_data)
                    if chunks_data:
                        print(f"✓ {filename}: {len(chunks_data)} chunks")
                    else:
                        print(f"⚠ {filename}: 0 chunks (empty or unreadable)")

        if not all_chunks:
            print("❌ No chunks extracted")
            return

        print(f"\n📊 Total chunks extracted: {len(all_chunks)}")
        print(f"⚡ Computing embeddings in batches of {batch_size}...")

        # Process embeddings and insertions in large batches
        for batch_start in range(0, len(all_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_chunks))
            batch = all_chunks[batch_start:batch_end]

            # Compute embeddings for batch (vectorized, very fast)
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=True,
            ).tolist()

            # Prepare batch for insertion
            documents = texts
            metadatas = [
                {"file": chunk["file"], "page": chunk["page"], "chunk": i}
                for i, chunk in enumerate(batch, batch_start)
            ]
            ids = [
                f"{chunk['file'].replace('.pdf', '')}_p{chunk['page']}_c{i}"
                for i, chunk in enumerate(batch, batch_start)
            ]

            # Insert batch to database
            self.collection.add(
                embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
            )

            progress = (batch_end / len(all_chunks)) * 100
            print(f"   Progress: {batch_end}/{len(all_chunks)} ({progress:.1f}%)")

        elapsed = time.time() - start_time
        chunks_per_sec = len(all_chunks) / elapsed

        print(f"\n✅ Indexing complete!")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Time: {elapsed:.1f}s ({chunks_per_sec:.1f} chunks/sec)")
        print(f"   Speed: ~{elapsed / len(pdf_files):.1f}s per PDF")

    def query_ollama(self, prompt, model="gemma2:27b", stream=True):
        """Query local Ollama model with streaming"""
        try:
            import requests
            import json

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": stream},
                stream=stream,
                timeout=180,
            )

            if stream:
                full_response = ""
                print("\n   ", end="", flush=True)
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            token = chunk["response"]
                            print(token, end="", flush=True)
                            full_response += token
                print("\n")
                return full_response
            else:
                result = response.json()
                return result.get("response", "")

        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Start it with: ollama serve"
        except requests.exceptions.Timeout:
            return "⚠️  Ollama took too long. Try:\n1. Use phi4 model\n2. Restart: killall ollama && ollama serve"
        except Exception as e:
            try:
                print(f"   Trying subprocess method...")
                result = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                return result.stdout.strip()
            except:
                return f"❌ Error: {str(e)}"

    def search(self, query, top_k=5, hybrid=True):
        """Search for relevant chunks - normalize query whitespace"""
        query_normalized = self._normalize_text(query)
        query_embedding = self.embedder.encode(query_normalized).tolist()

        initial_k = top_k * 3 if hybrid else top_k

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k,
        )

        if hybrid and results["documents"][0]:
            query_terms = set(query_normalized.lower().split())
            scored_results = []

            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                doc_lower = doc.lower()
                doc_terms = set(doc_lower.split())
                term_overlap = len(query_terms & doc_terms) / len(query_terms)
                phrase_bonus = 0.3 if query_normalized.lower() in doc_lower else 0

                proximity_bonus = 0
                if term_overlap > 0:
                    positions = []
                    for term in query_terms:
                        pos = doc_lower.find(term)
                        if pos != -1:
                            positions.append(pos)
                    if len(positions) == len(query_terms):
                        span = max(positions) - min(positions)
                        if span < 50:
                            proximity_bonus = 0.2

                hybrid_score = (
                    distance - (term_overlap * 0.4) - phrase_bonus - proximity_bonus
                )
                scored_results.append((hybrid_score, doc, metadata, distance))

            scored_results.sort(key=lambda x: x[0])

            results = {
                "documents": [[r[1] for r in scored_results[:top_k]]],
                "metadatas": [[r[2] for r in scored_results[:top_k]]],
                "distances": [[r[0] for r in scored_results[:top_k]]],
            }

        return results

    def search_with_diversity(self, query, top_k=5, chunks_per_pdf=2):
        """Search with diversity - ensure multiple PDFs are represented"""
        query_normalized = self._normalize_text(query)
        query_embedding = self.embedder.encode(query_normalized).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 5,
        )

        if not results["documents"][0]:
            return results

        selected = []
        pdf_counts = {}

        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            pdf = metadata["file"]
            count = pdf_counts.get(pdf, 0)

            if count < chunks_per_pdf:
                selected.append((doc, metadata, distance))
                pdf_counts[pdf] = count + 1

                if len(selected) >= top_k:
                    break

        if len(selected) < top_k:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if len(selected) >= top_k:
                    break
                if (doc, metadata, distance) not in selected:
                    selected.append((doc, metadata, distance))

        return {
            "documents": [[s[0] for s in selected]],
            "metadatas": [[s[1] for s in selected]],
            "distances": [[s[2] for s in selected]],
        }

    def debug_search(self, query, top_k=3):
        """Debug version that shows raw text and helps diagnose search issues"""
        query_normalized = self._normalize_text(query)
        print(f"\n🔍 Debug Search")
        print(f"   Original query: '{query}'")
        print(f"   Normalized query: '{query_normalized}'")

        results = self.search(query, top_k=20, hybrid=False)

        if not results["documents"][0]:
            print("\n❌ No results found")
            return

        print(
            f"\n📄 Top {min(len(results['documents'][0]), 10)} results (out of 20 retrieved):\n"
        )

        files_found = {}
        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0][:10],
                results["metadatas"][0][:10],
                results["distances"][0][:10],
            )
        ):
            filename = metadata["file"]
            if filename not in files_found:
                files_found[filename] = []
            files_found[filename].append((metadata["page"], distance))

            print(f"[Result {i + 1}] {metadata['file']}, Page {metadata['page']}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Text preview (first 150 chars):")
            print(f"   {doc[:150]}...")

            query_lower = query_normalized.lower()
            doc_lower = doc.lower()
            contains = query_lower in doc_lower
            print(f"   Contains exact query: {contains}")

            query_terms = set(query_normalized.lower().split())
            doc_terms = set(doc.lower().split())
            matching_terms = query_terms & doc_terms
            if matching_terms:
                print(f"   Matching terms: {matching_terms}")
            print()

        print("=" * 70)
        print(f"\n📊 Summary - PDFs in top 20 results:")
        for filename, pages_and_distances in sorted(files_found.items()):
            count = len(pages_and_distances)
            avg_distance = sum(d for _, d in pages_and_distances) / count
            pages = sorted(set(p for p, _ in pages_and_distances))
            print(f"   • {filename}: {count} chunks (avg distance: {avg_distance:.4f})")
            print(f"     Pages: {', '.join(map(str, pages))}")

        all_files = set()
        for metadata in results["metadatas"][0]:
            all_files.add(metadata["file"])

        print(f"\n📚 All unique PDFs in top 20 results: {len(all_files)}")
        for filename in sorted(all_files):
            if filename not in files_found:
                print(f"   • {filename} (appears only in positions 11-20)")

    def search_in_pdf(self, query, pdf_filename):
        """Search for query only within a specific PDF"""
        query_normalized = self._normalize_text(query)
        query_embedding = self.embedder.encode(query_normalized).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=100,
            where={"file": pdf_filename},
        )

        if not results["documents"][0]:
            print(f"\n❌ No results found in {pdf_filename}")
            print(f"\nℹ️  This PDF might:")
            print(f"   1. Not be indexed")
            print(f"   2. Not contain text matching '{query}'")
            return

        print(f"\n🔍 Results from {pdf_filename}:\n")
        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0][:5],
                results["metadatas"][0][:5],
                results["distances"][0][:5],
            )
        ):
            print(f"[Result {i + 1}] Page {metadata['page']}, Distance: {distance:.4f}")
            print(f"   {doc[:200]}...")
            print()

    def set_pdf_folder(self, folder_path):
        """Set the PDF folder for opening files"""
        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists():
            print(f"❌ Folder does not exist: {folder_path}")
            return False

        self.pdf_folder = folder

        config_file = self.data_dir / "chroma_db" / ".pdf_folder"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(str(folder))

        print(f"✅ PDF folder set to: {folder}")
        return True

    def open_pdf(self, filename, page=None):
        """Open PDF in Preview app, optionally at specific page"""
        if not self.pdf_folder:
            print("❌ PDF folder not set. Re-index PDFs first.")
            return

        pdf_path = self.pdf_folder / filename

        if not pdf_path.exists():
            print(f"❌ PDF not found: {pdf_path}")
            return

        print(f"📖 Opening {filename}" + (f" at page {page}" if page else "") + "...")

        try:
            subprocess.run(["open", "-a", "Preview", str(pdf_path)])
            print("✅ Opened in Preview")
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")

    def ask(
        self,
        question,
        top_k=10,
        model="gemma2:27b",
        show_sources=True,
        interactive=False,
        diverse=False,
        conversation_history=None,
        reuse_context=None,
    ):
        """Ask a question and get an answer"""

        if reuse_context:
            print(f"\n💭 Using context from previous search...")
            results = reuse_context
            context = reuse_context["context"]
            source_files = reuse_context["source_files"]
        else:
            print(f"\n🔍 Searching knowledge base...")

            if diverse:
                results = self.search_with_diversity(question, top_k)
            else:
                results = self.search(question, top_k)

            if not results["documents"][0]:
                print("❌ No relevant information found")
                return None

            context_parts = []
            source_files = {}

            for i, (doc, metadata) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                context_parts.append(
                    f"[Source {i + 1}: {metadata['file']}, Page {metadata['page']}]\n{doc}"
                )

                filename = metadata["file"]
                page = metadata["page"]
                if filename not in source_files:
                    source_files[filename] = []
                source_files[filename].append(page)

            context = "\n\n".join(context_parts)

        if show_sources and not reuse_context:
            print(
                f"\n📄 Found {len(results['documents'][0])} relevant chunks from {len(source_files)} file(s):"
            )
            for idx, (filename, pages) in enumerate(source_files.items(), 1):
                pages_str = ", ".join(map(str, sorted(set(pages))))
                print(f"   [{idx}] {filename} (Pages: {pages_str})")

        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for prev_q, prev_a in conversation_history[-3:]:
                history_text += f"Q: {prev_q}\nA: {prev_a}\n\n"

        prompt = f"""Based on the following context from PDF documents, answer the question. If the context doesn't contain enough information, say so.

Context:
{context}
{history_text}
Question: {question}

Answer:"""

        print(f"\n🤖 Generating answer with {model}...")

        answer = self.query_ollama(prompt, model)

        print("\n" + "=" * 70)
        print("📝 ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)

        context_data = {
            "context": context,
            "source_files": source_files,
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "distances": results["distances"],
        }

        if source_files:
            print("\n" + "─" * 70)
            print("📚 Source PDFs:")
            file_list = list(source_files.items())
            for idx, (filename, pages) in enumerate(file_list, 1):
                pages_str = ", ".join(map(str, sorted(set(pages))[:3]))
                if len(pages) > 3:
                    pages_str += "..."
                print(f"   [{idx}] {filename} (Pages: {pages_str})")

            print("\n💡 Options:")
            print("   • Enter number (1-{}) to open PDF".format(len(file_list)))
            print("   • Type 'q' to ask another question (keeps same context)")
            print("   • Type 'n' for new search")
            print("   • Press Enter to finish")

            try:
                choice = input("\nYour choice: ").strip().lower()

                if choice == "q":
                    return answer, results, True, (question, answer), context_data
                elif choice == "n":
                    return answer, results, True, (question, answer), None
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(file_list):
                        filename, pages = file_list[idx]
                        self.open_pdf(filename, min(pages))

                        print("\n" + "─" * 70)
                        print("Continue asking?")
                        print("  • Type 'y' to keep same context")
                        print("  • Type 'n' to start fresh")
                        print("  • Press Enter to finish")
                        cont = input("Your choice: ").strip().lower()
                        if cont == "y":
                            return (
                                answer,
                                results,
                                True,
                                (question, answer),
                                context_data,
                            )
                        elif cont == "n":
                            return answer, results, True, (question, answer), None
            except KeyboardInterrupt:
                print()

        return answer, results, False, (question, answer), None


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Quick search: rag <folder_path> <search_query>")
        print("  Index PDFs:   rag index <folder_path>")
        print("  Set folder:   rag setfolder <folder_path>")
        print("  List indexed: rag list")
        print("  Ask question: rag ask '<question>'")
        print("  Debug search: rag debug '<query>'")
        print("  Interactive:  rag interactive")
        print("\nExamples:")
        print('  rag ~/Documents/PDFs "majority element"')
        print("  rag index ~/Documents/Textbooks")
        print("  rag setfolder ~/Documents/Textbooks")
        print("  rag list")
        print("  rag ask 'What is dynamic programming?'")
        print("  rag debug 'dynamic programming'")
        print("  rag interactive")
        sys.exit(1)

    # Check if this is a quick search command
    if len(sys.argv) == 3:
        first_arg = sys.argv[1]
        second_arg = sys.argv[2]

        potential_path = Path(first_arg).expanduser().resolve()
        known_commands = ["index", "ask", "debug", "list", "setfolder", "interactive"]

        if potential_path.exists() and first_arg not in known_commands:
            pdf_folder = first_arg
            query = second_arg

            folder = Path(pdf_folder).expanduser().resolve()
            collection_name = f"pdfs_{folder.name}".replace(" ", "_").replace("-", "_")

            print(f"🔍 Quick Search Mode")
            print(f"   Folder: {folder}")
            print(f"   Query: '{query}'")
            print()

            rag = LocalRAG(collection_name=collection_name, pdf_folder=folder)

            if rag.collection.count() == 0:
                print("📝 No existing index found - indexing PDFs...")
                rag.index_pdfs(folder)
                print()
            else:
                print(f"✅ Using existing index ({rag.collection.count()} chunks)")
                print()

            conversation_history = []
            current_context = None
            result = rag.ask(query, conversation_history=conversation_history)

            if result and len(result) >= 3 and result[2]:
                if len(result) >= 4:
                    conversation_history.append(result[3])
                if len(result) >= 5:
                    current_context = result[4]

                print("\n" + "=" * 70)
                print("🔄 Continue asking questions (type 'quit' to exit)")
                print("=" * 70)

                while True:
                    try:
                        follow_up = input("\n💬 Next question: ").strip()

                        if follow_up.lower() in ["quit", "exit"]:
                            print("👋 Goodbye!")
                            break

                        if not follow_up:
                            continue

                        result = rag.ask(
                            follow_up,
                            conversation_history=conversation_history,
                            reuse_context=current_context,
                        )

                        if result and len(result) >= 4:
                            conversation_history.append(result[3])
                            if len(conversation_history) > 5:
                                conversation_history = conversation_history[-5:]

                        if result and len(result) >= 5:
                            if result[4] is not None:
                                current_context = result[4]
                            else:
                                current_context = None

                        if not (result and len(result) >= 3 and result[2]):
                            break

                    except KeyboardInterrupt:
                        print("\n👋 Goodbye!")
                        break

            return

    command = sys.argv[1].lower()

    rag = LocalRAG()

    if command == "index":
        if len(sys.argv) < 3:
            print("❌ Please provide folder path")
            print("Usage: rag index <folder_path>")
            sys.exit(1)

        folder = sys.argv[2]
        rag.index_pdfs(folder)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("❌ Please provide a question")
            print("Usage: rag ask '<question>'")
            sys.exit(1)

        question = sys.argv[2]
        result = rag.ask(question)

        if result:
            answer = result[0] if len(result) >= 1 else None

            print("\n" + "=" * 70)
            print("📝 ANSWER:")
            print("=" * 70)
            if answer:
                print(answer)
            print("=" * 70)

    elif command == "debug":
        if len(sys.argv) < 3:
            print("❌ Please provide a query to debug")
            print("Usage: rag debug '<query>' [pdf_filename]")
            print("\nExamples:")
            print("  rag debug 'majority element'")
            print("  rag debug 'majority element' 'mybook.pdf'")
            sys.exit(1)

        query = sys.argv[2]

        if len(sys.argv) > 3:
            pdf_filename = sys.argv[3]
            rag.search_in_pdf(query, pdf_filename)
        else:
            rag.debug_search(query)

    elif command == "interactive":
        print("\n" + "=" * 70)
        print("🎯 Interactive RAG Mode")
        print("=" * 70)
        print("Type your questions (or 'quit' to exit)")
        print("Type 'debug: <query>' to see debug info for a search")
        print("Type 'list' to see all indexed PDFs")
        print("Type 'setfolder <path>' to set PDF folder location")
        print("")

        conversation_history = []
        current_context = None

        while True:
            try:
                question = input("\n💬 Question: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                if question.lower() == "list":
                    rag.list_indexed_pdfs()
                    continue

                if question.lower().startswith("setfolder "):
                    folder = question[10:].strip()
                    rag.set_pdf_folder(folder)
                    continue

                if question.lower().startswith("debug:"):
                    parts = question[6:].strip().split(" in ", 1)
                    query = parts[0].strip()

                    if len(parts) > 1:
                        pdf_filename = parts[1].strip()
                        rag.search_in_pdf(query, pdf_filename)
                    else:
                        rag.debug_search(query)
                    continue

                result = rag.ask(
                    question,
                    conversation_history=conversation_history,
                    reuse_context=current_context,
                )

                if result and len(result) >= 4:
                    conversation_history.append(result[3])
                    if len(conversation_history) > 5:
                        conversation_history = conversation_history[-5:]

                if result and len(result) >= 5:
                    if result[4] is not None:
                        current_context = result[4]
                    else:
                        current_context = None

                continue

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

    elif command == "list":
        rag.list_indexed_pdfs()

    elif command == "setfolder":
        if len(sys.argv) < 3:
            print("❌ Please provide folder path")
            print("Usage: rag setfolder <folder_path>")
            sys.exit(1)

        folder = sys.argv[2]
        rag.set_pdf_folder(folder)

    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: index, ask, debug, list, setfolder, interactive")


if __name__ == "__main__":
    main()
