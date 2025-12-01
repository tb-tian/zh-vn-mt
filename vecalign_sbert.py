"""
LaBSE + Vecalign Sentence Alignment for Chinese-Vietnamese MT Corpus

Combines:
- LaBSE (Language-agnostic BERT Sentence Embedding) for cross-lingual embeddings
- Vecalign algorithm for sophisticated many-to-many alignment

This handles 1:1, 1:2, 2:1, 2:2 alignments and deletions.

Install: pip install sentence-transformers
"""

import json
import re
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pysbd

@dataclass
class AlignedPair:
    zh: str
    vi: str
    similarity: float
    alignment_type: str  # e.g., "1-1", "2-1", "1-2"
    source: str
    source_id: int


class VecalignSBERT:
    """
    Vecalign-style alignment using LaBSE embeddings.
    
    Supports alignment types: 0-1, 1-0, 1-1, 1-2, 2-1, 2-2
    """
    
    # Alignment types: (num_src, num_tgt)
    ALIGNMENT_TYPES = [
        (1, 0),  # deletion (zh only)
        (0, 1),  # insertion (vi only)
        (1, 1),  # 1-to-1
        (1, 2),  # 1-to-2
        (2, 1),  # 2-to-1
        (2, 2),  # 2-to-2
    ]
    
    def __init__(
        self, 
        similarity_threshold: float = 0.5,
        deletion_penalty: float = 0.3,
        model_name: str = "sentence-transformers/LaBSE"
    ):
        """
        Initialize Vecalign-style aligner with LaBSE.
        
        Args:
            similarity_threshold: Minimum similarity for valid alignment
            deletion_penalty: Cost for skipping a sentence (0-1 or 1-0)
            model_name: Sentence-transformers model
        """
        from sentence_transformers import SentenceTransformer
        import torch
        
        self.similarity_threshold = similarity_threshold
        self.deletion_penalty = deletion_penalty
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seg_zh = pysbd.Segmenter(language="zh", clean=False)
        self.seg_en = pysbd.Segmenter(language="en", clean=False)
        
        print(f"Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Model loaded!")
    
    # def split_chinese_sentences(self, text: str) -> List[str]:
    #     """Split Chinese text into sentences."""
    #     text = re.sub(r'\n+', '\n', text)
    #     # Chinese full-width and half-width punctuation
    #     pattern = r'(?<=[。！？；!?｡…])[」』"）\)]?\s*|\n'
    #     sentences = re.split(pattern, text)
    #     return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
    
    # def split_vietnamese_sentences(self, text: str) -> List[str]:
    #     """Split Vietnamese text into sentences."""
    #     text = re.sub(r'\n+', '\n', text)
    #     pattern = r'(?<=[.!?])\s+|\n'
    #     sentences = re.split(pattern, text)
    #     return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def split_chinese_sentences(self, text: str) -> List[str]:
        """Split Chinese text into sentences."""
        sentences = self.seg_zh.segment(text)
        # Filter out very short noise, but keep 2-char sentences like "好的。"
        return [s.strip() for s in sentences if len(s.strip()) > 1]
    
    def split_vietnamese_sentences(self, text: str) -> List[str]:
        """Split Vietnamese text into sentences."""
        sentences = self.seg_en.segment(text)
        return [s.strip() for s in sentences if len(s.strip()) > 1]
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences to embeddings."""
        if not sentences:
            return np.array([])
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    
    def merge_embeddings(self, embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
        """Merge multiple sentence embeddings by averaging."""
        if len(indices) == 0:
            return np.zeros(embeddings.shape[1])
        if len(indices) == 1:
            return embeddings[indices[0]]
        return np.mean(embeddings[indices], axis=0)
    
    def merge_sentences(self, sentences: List[str], indices: List[int]) -> str:
        """Merge multiple sentences into one."""
        return ' '.join(sentences[i] for i in indices)
    
    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    
    def alignment_cost(
        self,
        zh_embs: np.ndarray,
        vi_embs: np.ndarray,
        zh_indices: List[int],
        vi_indices: List[int]
    ) -> Tuple[float, float]:
        """
        Compute alignment cost and similarity for a candidate alignment.
        
        Returns:
            (cost, similarity): Lower cost is better, similarity is for output
        """
        # Deletion cases
        if len(zh_indices) == 0 or len(vi_indices) == 0:
            num_deleted = len(zh_indices) + len(vi_indices)
            return (self.deletion_penalty * num_deleted, 0.0)
        
        # Merge embeddings if multiple sentences
        zh_emb = self.merge_embeddings(zh_embs, zh_indices)
        vi_emb = self.merge_embeddings(vi_embs, vi_indices)
        
        similarity = self.cosine_similarity(zh_emb, vi_emb)
        
        # Cost = 1 - similarity (so higher similarity = lower cost)
        # Add small penalty for many-to-many alignments to prefer simpler alignments
        complexity_penalty = 0.05 * (len(zh_indices) + len(vi_indices) - 2)
        cost = (1.0 - similarity) + complexity_penalty
        
        return (cost, similarity)
    
    def vecalign_dp(
        self,
        zh_sents: List[str],
        vi_sents: List[str],
        zh_embs: np.ndarray,
        vi_embs: np.ndarray
    ) -> List[Tuple[List[int], List[int], float, str]]:
        """
        Vecalign-style dynamic programming alignment.
        
        Returns list of (zh_indices, vi_indices, similarity, alignment_type)
        """
        n, m = len(zh_sents), len(vi_sents)
        
        if n == 0 or m == 0:
            return []
        
        # DP table: dp[i][j] = minimum cost to align zh[:i] with vi[:j]
        INF = float('inf')
        dp = np.full((n + 1, m + 1), INF)
        dp[0, 0] = 0.0
        
        # Backtrack: stores (prev_i, prev_j, zh_indices, vi_indices, similarity, type)
        backtrack = [[None for _ in range(m + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            for j in range(m + 1):
                if dp[i, j] == INF:
                    continue
                
                for (di, dj) in self.ALIGNMENT_TYPES:
                    ni, nj = i + di, j + dj
                    
                    if ni > n or nj > m:
                        continue
                    
                    # Get indices for this alignment
                    zh_idx = list(range(i, ni))
                    vi_idx = list(range(j, nj))
                    
                    cost, sim = self.alignment_cost(zh_embs, vi_embs, zh_idx, vi_idx)
                    
                    # Skip low-similarity alignments (except deletions)
                    if di > 0 and dj > 0 and sim < self.similarity_threshold:
                        continue
                    
                    new_cost = dp[i, j] + cost
                    
                    if new_cost < dp[ni, nj]:
                        dp[ni, nj] = new_cost
                        align_type = f"{di}-{dj}"
                        backtrack[ni][nj] = (i, j, zh_idx, vi_idx, sim, align_type)
        
        # Backtrack to get alignments
        alignments = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if backtrack[i][j] is None:
                break
            
            prev_i, prev_j, zh_idx, vi_idx, sim, align_type = backtrack[i][j]
            
            # Only keep actual alignments (not pure deletions)
            if len(zh_idx) > 0 and len(vi_idx) > 0:
                alignments.append((zh_idx, vi_idx, sim, align_type))
            
            i, j = prev_i, prev_j
        
        alignments.reverse()
        return alignments
    
    def align_document(self, zh_text: str, vi_text: str) -> List[Tuple[str, str, float, str]]:
        """
        Align sentences from a document pair using Vecalign algorithm.
        
        Returns list of (zh_sentence, vi_sentence, similarity, alignment_type)
        """
        zh_sents = self.split_chinese_sentences(zh_text)
        vi_sents = self.split_vietnamese_sentences(vi_text)
        
        if not zh_sents or not vi_sents:
            return []
        
        # Encode all sentences
        zh_embs = self.encode_sentences(zh_sents)
        vi_embs = self.encode_sentences(vi_sents)
        
        # Run Vecalign DP
        alignments = self.vecalign_dp(zh_sents, vi_sents, zh_embs, vi_embs)
        
        # Convert indices to sentences
        results = []
        for zh_idx, vi_idx, sim, align_type in alignments:
            zh_merged = self.merge_sentences(zh_sents, zh_idx)
            vi_merged = self.merge_sentences(vi_sents, vi_idx)
            results.append((zh_merged, vi_merged, sim, align_type))
        
        return results


def process_data(input_path: str, output_prefix: str, threshold: float = 0.5):
    """Process data.json and create aligned corpus using Vecalign."""
    print(f"\n{'='*60}")
    print("LaBSE + Vecalign Sentence Alignment")
    print(f"{'='*60}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} documents")
    
    aligner = VecalignSBERT(similarity_threshold=threshold)
    all_pairs: List[AlignedPair] = []
    
    # Statistics for alignment types
    type_counts = {}
    
    for item in tqdm(data, desc="Aligning"):
        if 'zh' not in item or 'vi' not in item:
            continue
        
        aligned = aligner.align_document(item['zh'], item['vi'])
        
        for zh, vi, sim, align_type in aligned:
            all_pairs.append(AlignedPair(
                zh=zh, vi=vi, similarity=float(sim),
                alignment_type=align_type,
                source=item.get('source', 'unknown'),
                source_id=item.get('id', 0)
            ))
            type_counts[align_type] = type_counts.get(align_type, 0) + 1
    
    print(f"\nTotal aligned pairs: {len(all_pairs)}")
    
    # Save outputs
    with open(f"{output_prefix}.zh", 'w', encoding='utf-8') as fz, \
         open(f"{output_prefix}.vi", 'w', encoding='utf-8') as fv:
        for p in all_pairs:
            fz.write(p.zh + '\n')
            fv.write(p.vi + '\n')
    
    with open(f"{output_prefix}.json", 'w', encoding='utf-8') as f:
        json.dump([asdict(p) for p in all_pairs], f, ensure_ascii=False, indent=2)
    
    with open(f"{output_prefix}.tsv", 'w', encoding='utf-8') as f:
        f.write("zh\tvi\tsimilarity\talignment_type\tsource\tsource_id\n")
        for p in all_pairs:
            zh_clean = p.zh.replace('\t', ' ').replace('\n', ' ')
            vi_clean = p.vi.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{zh_clean}\t{vi_clean}\t{p.similarity:.4f}\t{p.alignment_type}\t{p.source}\t{p.source_id}\n")
    
    print(f"\nSaved to: {output_prefix}.zh, .vi, .json, .tsv")
    
    # Stats
    if all_pairs:
        sims = [p.similarity for p in all_pairs]
        print(f"\nStatistics:")
        print(f"  Total pairs: {len(all_pairs)}")
        print(f"  Similarity: min={min(sims):.3f}, max={max(sims):.3f}, mean={np.mean(sims):.3f}")
        print(f"\nAlignment type distribution:")
        for atype, count in sorted(type_counts.items()):
            pct = 100 * count / len(all_pairs)
            print(f"  {atype}: {count} ({pct:.1f}%)")
    
    return all_pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LaBSE + Vecalign sentence alignment")
    parser.add_argument("--input", default="subsubset.json", help="Input JSON file")
    parser.add_argument("--output", default="corpus_vecalign_scratch", help="Output prefix")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    args = parser.parse_args()
    
    process_data(args.input, args.output, args.threshold)
