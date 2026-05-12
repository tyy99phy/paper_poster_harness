from .assets import (
    apply_detections_to_spec,
    apply_figure_selection_to_spec,
    build_assets_manifest,
    infer_asset_roots,
)
from .arxiv import download_arxiv_bundle, resolve_arxiv_with_llm
from .config import load_harness_config, write_default_harness_config
from .account_auth import load_account_auth, find_account_auth_file, list_account_auth_files
from .auth_login import run_browser_login
from .image_backend import generate_images_from_config
from .layout_contract import build_layout_contract
from .llm import ChatGPTAccountResponsesProvider, extract_json_from_response, extract_json_from_text
from .llm_stages import (
    copy_deck_from_text,
    critique_poster_template,
    detect_placeholders_from_image,
    draft_spec_from_text,
    physics_quiz_from_text,
    qa_poster,
    select_figures,
    storyboard_from_text,
)
from .schemas import (
    copy_deck_schema,
    default_poster_spec,
    figure_selection_schema,
    placeholder_detection_schema,
    physics_quiz_schema,
    poster_qa_schema,
    poster_template_critic_schema,
    poster_spec_schema,
    schema_skeleton,
    storyboard_schema,
)

__all__ = [
    "ChatGPTAccountResponsesProvider",
    "load_account_auth",
    "find_account_auth_file",
    "list_account_auth_files",
    "generate_images_from_config",
    "build_layout_contract",
    "run_browser_login",
    "load_harness_config",
    "write_default_harness_config",
    "resolve_arxiv_with_llm",
    "download_arxiv_bundle",
    "build_assets_manifest",
    "infer_asset_roots",
    "apply_figure_selection_to_spec",
    "apply_detections_to_spec",
    "extract_json_from_response",
    "extract_json_from_text",
    "draft_spec_from_text",
    "storyboard_from_text",
    "physics_quiz_from_text",
    "copy_deck_from_text",
    "select_figures",
    "critique_poster_template",
    "detect_placeholders_from_image",
    "qa_poster",
    "default_poster_spec",
    "poster_spec_schema",
    "storyboard_schema",
    "physics_quiz_schema",
    "copy_deck_schema",
    "figure_selection_schema",
    "placeholder_detection_schema",
    "poster_qa_schema",
    "poster_template_critic_schema",
    "schema_skeleton",
]
