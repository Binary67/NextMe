from graphtool.corpus import synchronize_documents
from graphtool.ingestion import load_audio_transcription_terms, load_documents
from graphtool.llm import load_azure_openai_config
from graphtool.run_logging import configure_run_logger
from graphtool.runtime import DEFAULT_MAX_LOG_FILES, create_runtime, default_paths
from graphtool.visualization import export_knowledge_base_visualizations


def main() -> None:
    paths = default_paths()
    logger = configure_run_logger(paths.logs_dir, DEFAULT_MAX_LOG_FILES)
    logger.info("Started GraphTool ingestion")

    try:
        config = load_azure_openai_config()
        runtime = create_runtime(config, paths=paths)
        audio_transcription_terms = load_audio_transcription_terms(
            runtime.paths.audio_transcription_glossary_path
        )
        documents = load_documents(
            runtime.paths.documents_dir,
            source_root=runtime.paths.root,
            pdf_llm=runtime.fast_llm,
            pdf_cache_dir=runtime.paths.pdf_conversions_dir,
            presentation_cache_dir=runtime.paths.presentation_conversions_dir,
            audio_transcriber=runtime.audio_transcriber,
            audio_corrector=runtime.fast_llm,
            audio_cache_dir=runtime.paths.audio_transcriptions_dir,
            audio_transcription_terms=audio_transcription_terms,
        )
        logger.info(
            "Loaded %s %s",
            len(documents),
            "document" if len(documents) == 1 else "documents",
        )

        sync_result = synchronize_documents(
            documents,
            runtime.corpus_stores,
            runtime.fast_llm,
            chunk_extraction_store=runtime.chunk_extraction_store,
            dropped_edges_path=runtime.paths.dropped_edges_path,
            min_candidate_similarity=(
                config.entity_resolution_min_candidate_similarity
            ),
        )
        logger.info(
            "Synchronization complete: %s added, %s changed, %s removed, "
            "%s unchanged",
            len(sync_result.added_sources),
            len(sync_result.changed_sources),
            len(sync_result.deleted_sources),
            len(sync_result.unchanged_sources),
        )

        logger.info("Exporting visualizations")
        visualization_paths = export_knowledge_base_visualizations(
            runtime.graph_store,
            runtime.paths.visualizations_dir,
            knowledge_base_store=runtime.knowledge_base_store,
        )
        logger.info(
            "Exported %s %s",
            len(visualization_paths),
            "visualization" if len(visualization_paths) == 1 else "visualizations",
        )

        print("Visualizations:")
        for path in visualization_paths:
            print(f"- {path}")

        logger.info("Finished GraphTool ingestion")
    except Exception:
        logger.exception("Ingestion failed")
        raise


if __name__ == "__main__":
    main()
