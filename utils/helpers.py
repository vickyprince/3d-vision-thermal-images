def get_pointmap(pred):
    for key in ["pts3d", "pointmap", "pointmaps", "predicted_pts3d", "pts3d_in_other_view"]:
        if key in pred:
            print(f"Using key '{key}' for pointmap.")
            return pred[key]
    raise KeyError("No recognized pointmap key found. Keys: " + str(list(pred.keys())))