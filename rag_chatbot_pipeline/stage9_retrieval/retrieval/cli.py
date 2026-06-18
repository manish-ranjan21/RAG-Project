"""
CLI entry point for the retrieval service.

Usage:
    # Basic vector search (matches your original Stage 9)
    python -m retrieval.cli "what is feature engineering?"
    
    # More results
    python -m retrieval.cli "what is feature engineering?" --k 10
    
    # Hybrid (vector + keyword)
    python -m retrieval.cli "FINRA Rule 2111 risk tolerance" --mode hybrid
    
    # With reranking
    python -m retrieval.cli "what is feature engineering?" --rerank
    
    # JSON output (for piping to jq)
    python -m retrieval.cli "what is feature engineering?" --json
    
    # Test mode (uses mock embedder, no API calls)
    python -m retrieval.cli "test query" --mock
    
    # Health check only
    python -m retrieval.cli --health-check
    
    # Show cache stats
    python -m retrieval.cli "query" --show-stats
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .bootstrap import bootstrap
from .logging_setup import log_query
from .service import RetrievalRequest


def format_result_text(result, idx: int) -> str:
    """Pretty-print a single result for terminal viewing."""
    lines = []
    
    score_parts = []
    if result.vector_similarity is not None:
        score_parts.append(f"vec={result.vector_similarity:.4f}")
    if result.keyword_score is not None:
        score_parts.append(f"kw={result.keyword_score:.4f}")
    if result.combined_score is not None:
        score_parts.append(f"combined={result.combined_score:.4f}")
    if result.rerank_score is not None:
        score_parts.append(f"rerank={result.rerank_score:.4f}")
    score_str = "  ".join(score_parts)
    
    lines.append(f"[{idx}] {score_str}")
    lines.append(f"    chunk_id:  {result.chunk_id}")
    lines.append(f"    heading:   {result.section_heading or '(none)'}")
    if result.section_path:
        path = result.section_path
        if len(path) > 80:
            path = path[:77] + "..."
        lines.append(f"    path:      {path}")
    if result.page_range:
        lines.append(f"    pages:     {result.page_range[0]}-{result.page_range[1]}")
    
    preview = result.text.strip().replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:237] + "..."
    lines.append(f"    preview:   {preview}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Production-grade Stage 9 retrieval CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("question", nargs="?", help="Your query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"],
                       default="hybrid", help="Search mode (default: hybrid)")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=True,
                       help="Apply cross-encoder reranking (on by default)")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                       help="Disable reranking (faster, lower precision)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--mock", action="store_true", 
                       help="Use mock embedder (no API key required)")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                       help="Path to .env file")
    parser.add_argument("--health-check", action="store_true",
                       help="Run health check and exit")
    parser.add_argument("--show-stats", action="store_true",
                       help="Show cache stats after query")
    parser.add_argument("--min-similarity", type=float, default=0.0,
                       help="Minimum similarity threshold")
    parser.add_argument("--access-class", action="append", default=None,
                       help="Access classifications (repeatable)")
    parser.add_argument("--access-group", action="append", default=None,
                       help="Access groups (repeatable)")
    parser.add_argument("--doc-type", action="append", default=None,
                       help="Doc types to include (repeatable)")
    parser.add_argument("--ef-search", type=int, default=None,
                       help="Override HNSW ef_search")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable embedding cache")
    
    args = parser.parse_args()
    
    if not args.question and not args.health_check:
        parser.error("question is required unless --health-check is set")
    
    if args.mock:
        os.environ["EMBEDDING_PROVIDER"] = "mock"
        os.environ["EMBEDDING_MODEL"] = "mock-embedder"
        # Offline mode: use the lexical reranker so we don't load torch/the model.
        os.environ["RERANK_PROVIDER"] = "heuristic"
    
    if args.no_cache:
        os.environ["CACHE_ENABLED"] = "false"
    
    try:
        app = bootstrap(env_file=args.env_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.health_check:
            from .database import CorpusHealthCheck
            health = CorpusHealthCheck(
                app.pool, app.config.embedding.model_name, 
                app.config.embedding.dimension
            )
            result = health.run()
            print(json.dumps(result, indent=2))
            return
        
        request = RetrievalRequest(
            query=args.question,
            top_k=args.k,
            search_mode=args.mode,
            rerank=args.rerank,
            min_similarity=args.min_similarity,
            hnsw_ef_search=args.ef_search,
        )
        if args.access_class:
            request.access_classifications = args.access_class
        if args.access_group:
            request.access_groups = args.access_group
        if args.doc_type:
            request.doc_types = args.doc_type
        
        with log_query(args.question) as ctx:
            results = app.service.retrieve(request, ctx)
        
        app.metrics.record_query(ctx)
        
        if args.json:
            output = {
                "query": args.question,
                "mode": args.mode,
                "duration_ms": int(ctx.total_duration_seconds * 1000),
                "results_count": len(results),
                "cache_hit": ctx.cache_hit,
                "results": [r.to_dict() for r in results]
            }
            print(json.dumps(output, indent=2))
        else:
            print()
            print("=" * 78)
            print(f"QUERY: {args.question}")
            print(f"MODE:  {args.mode}{' (with rerank)' if args.rerank else ''}")
            print(f"TIME:  {ctx.total_duration_seconds*1000:.0f}ms total "
                 f"(embed: {(ctx.embed_duration_seconds or 0)*1000:.0f}ms, "
                 f"sql: {(ctx.sql_duration_seconds or 0)*1000:.0f}ms"
                 f"{', rerank: ' + str(int((ctx.rerank_duration_seconds or 0)*1000)) + 'ms' if args.rerank else ''}"
                 f"){'  [CACHE HIT]' if ctx.cache_hit else ''}")
            if ctx.estimated_cost_usd > 0:
                print(f"COST:  ${ctx.estimated_cost_usd:.6f} (this query)")
            print("=" * 78)
            
            if not results:
                print("\nNo results matched. Possible causes:")
                print("- Corpus is empty (run Stage 8 load first)")
                print("- min_similarity threshold too high (--min-similarity 0)")
                print("- Permission filter excluded everything (check --access-class/--access-group)")
            else:
                for i, result in enumerate(results, 1):
                    print()
                    print(format_result_text(result, i))
        
        if args.show_stats:
            print()
            print("-" * 78)
            print("CACHE STATS:", json.dumps(app.cache.stats(), indent=2))
            print("POOL HEALTH:", json.dumps(app.pool.health_check(), indent=2))
    
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
