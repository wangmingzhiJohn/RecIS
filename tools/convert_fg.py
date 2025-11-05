import argparse
import json


def update_features(
    features,
    readonly_features=None,
    not_trainables=None,
    hash_type_map=None,
    emb_type_map=None,
    emb_device_map=None,
):
    for feature in features:
        if "features" not in feature:
            if "boundaries" in feature:
                feature["gen_key_type"] = "boundary"
                feature["gen_val_type"] = "lookup"
            elif "hash_bucket_size" in feature:
                feature["gen_key_type"] = "hash"
                feature["gen_val_type"] = "lookup"
                if "compress_strategy" in feature:
                    feature["gen_key_type"] = "multihash"
                    feature["gen_val_type"] = "multihash_lookup"
            else:
                feature["gen_key_type"] = "idle"
                feature["gen_val_type"] = "idle"

            if (
                readonly_features is not None
                and feature["feature_name"] in readonly_features
            ):
                feature["admit_hook"] = {"name": "ReadOnly"}
            if not_trainables is not None and feature["feature_name"] in not_trainables:
                feature["trainable"] = False
            if hash_type_map is not None and feature["feature_name"] in hash_type_map:
                feature["hash_type"] = hash_type_map[feature["feature_name"]]
            if emb_type_map is not None and feature["feature_name"] in emb_type_map:
                feature["emb_type"] = emb_type_map[feature["feature_name"]]
            if emb_device_map is not None and feature["feature_name"] in emb_device_map:
                feature["emb_device"] = emb_device_map[feature["feature_name"]]

        if "features" in feature:
            update_features(feature["features"])


def process_json(
    input_file,
    output_file,
    readonly_features=None,
    not_trainables=None,
    hash_type_map=None,
    emb_type_map=None,
    emb_device_map=None,
):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("xdl", []):
        data["xdl"] = False

    update_features(
        data.get("features", []),
        readonly_features,
        not_trainables,
        hash_type_map,
        emb_type_map,
        emb_device_map,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert V2 ckpt to V3.")
    parser.add_argument("--src_path", type=str, required=True, help="Input fg path")
    parser.add_argument("--dst_path", type=str, required=True, help="Output fg path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_json(args.src_path, args.dst_path)
