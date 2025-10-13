# env_check.py
"""Inspect OCI Generative AI configuration and list available models/providers.

Usage examples:
  python env_check.py                    # rely on env vars
  python env_check.py --compartment <ocid> --model-id <ocid>  # override via flags
"""

import argparse
import os
import sys

import oci

parser = argparse.ArgumentParser(description="Inspect OCI Generative AI model configuration")
parser.add_argument("--profile", default=os.getenv("OCI_CONFIG_PROFILE", "DEFAULT"), help="OCI config profile (defaults to env OCI_CONFIG_PROFILE or DEFAULT)")
parser.add_argument("--config", default=os.path.expanduser(os.getenv("OCI_CONFIG_PATH", "~/.oci/config")), help="Path to OCI config file")
parser.add_argument("--model-id", dest="model_id", default=os.getenv("OCI_GENAI_MODEL_ID"), help="Specific model OCID to describe")
parser.add_argument("--compartment", dest="compartment_id", default=os.getenv("OCI_COMPARTMENT_ID"), help="Compartment OCID used to list available models")
args = parser.parse_args()

if not args.compartment_id:
    sys.exit("Missing compartment OCID. Set OCI_COMPARTMENT_ID or pass --compartment.")

try:
    config = oci.config.from_file(args.config, args.profile)
except Exception as exc:
    sys.exit(f"Unable to load OCI config ({args.config}, profile {args.profile}): {exc}")

mgmt = oci.generative_ai.GenerativeAiClient(config=config)

if args.model_id:
    print("=== CURRENT MODEL (get_model) ===")
    try:
        mdl = mgmt.get_model(model_id=args.model_id).data
    except Exception as exc:
        print(f"Failed to fetch model {args.model_id}: {exc}")
    else:
        vendor = getattr(mdl, "vendor", None) or getattr(mdl, "provider", None) or getattr(mdl, "provider_name", None)
        print("id:           ", mdl.id)
        print("vendor:       ", vendor)
        print("display_name: ", getattr(mdl, "display_name", None))
        print("version:      ", getattr(mdl, "version", None))
        print("state:        ", getattr(mdl, "lifecycle_state", None))
    print()
else:
    print("No OCI_GENAI_MODEL_ID configured; skipping get_model lookup.\n")

print("=== AVAILABLE MODELS IN COMPARTMENT ===")
try:
    response = mgmt.list_models(compartment_id=args.compartment_id)
except Exception as exc:
    sys.exit(f"list_models failed: {exc}")

models = getattr(response.data, "items", response.data)
if not models:
    print("(none returned)")
else:
    for model in models:
        vendor = getattr(model, "vendor", None) or getattr(model, "provider", None) or getattr(model, "provider_name", None)
        display_name = getattr(model, "display_name", None)
        version = getattr(model, "version", None)
        print(f"{model.id} | {vendor} | {display_name} | {version}")
