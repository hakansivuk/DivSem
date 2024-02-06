import os

# Utility class for the demo
class AppUtils():
    _label_id_map = {
        0: {
            "Sky": 2,
            "Tree": 4,
            "Road": 52,
        },
        1: {
            "Sky": 2,
            "Tree": 4,
            "Mountain": 16,
            "Water": 21,
        },
        2: {
            "Sky": 2,
            "Mountain": 16,
        },
        3: {
            "Sky": 2,
            "Ground": 13,
            "Mountain": 16,
        },
        4: {
            "Sky": 2,
            "Mountain": 16,
        },
        5: {
            "Sky": 2,
            "Mountain": 16,
        },
        6: {
            "Sky": 2,
            "Tree": 4,
            "Mountain": 16,
        },
    }
    _inst_id_map = {
        0: {
            "Sky": 362,
            "Tree": 604,
            "Cim": 2056,
            "Road": 6412,
        },
        1: {
            "Sky": 362,
            "Tree": 604,
            "Mountain": 2056,
            "Water": 2661,
        },
        2: {
            "Sky": 362,
            "Mountain": 2056,
        },
        3: {
            "Sky": 362,
            "Ground": 1693,
            "Mountain": 2056,
        },
        4: {
            "Sky": 362,
            "Mountain": 2056,
        },
        5: {
            "Sky": 362,
            "Mountain": 2056,
        },
        6: {
            "Sky": 362,
            "Tree": 604,
            "Mountain": 2056,
        },
    }
        
    _save_paths = {
        "image": "gradio_files/samples/test_processed/images",
        "labels": "gradio_files/samples/test_processed/labels",
        "inst_map": "gradio_files/samples/test_processed/inst_map",
        "predefined_masks": "gradio_files/samples/test_processed/predefined_masks/type_0",
        "synthesized_image": "gradio_files/samples/synthesized_image"
    }
    
    @staticmethod
    def clear():
        for save_path in AppUtils._save_paths.values():
            AppUtils._create_folder(save_path)

        os.system("rm -rf gradio_files/samples/test_processed/images/*")
        os.system("rm -rf gradio_files/samples/test_processed/labels/*")
        os.system("rm -rf gradio_files/samples/test_processed/inst_map/*")
        os.system("rm -rf gradio_files/samples/test_processed/predefined_masks/type_0/*")

    @staticmethod
    def get_examples():
        return [
            [0, "gradio_files/samples/flickr-landscape/images/832-41253531765_83c1767ba9_o.png", "gradio_files/samples/flickr-landscape/colored/832-41253531765_83c1767ba9_o.png"],
            [1, "gradio_files/samples/flickr-landscape/images/3736-9818172074_156d4682f3_o.png", "gradio_files/samples/flickr-landscape/colored/3736-9818172074_156d4682f3_o.png"],
            [2, "gradio_files/samples/flickr-landscape/images/7343-9965972016_a822e52102_o.png", "gradio_files/samples/flickr-landscape/colored/7343-9965972016_a822e52102_o.png"],
            [3, "gradio_files/samples/flickr-landscape/images/7503-16108428460_622fcdb3ca_o.png", "gradio_files/samples/flickr-landscape/colored/7503-16108428460_622fcdb3ca_o.png"],
            [4, "gradio_files/samples/flickr-landscape/images/7921-47167099321_02f96ba4f6_o.png", "gradio_files/samples/flickr-landscape/colored/7921-47167099321_02f96ba4f6_o.png"],
            [5, "gradio_files/samples/flickr-landscape/images/8016-7167270731_b9843b1072_o.png", "gradio_files/samples/flickr-landscape/colored/8016-7167270731_b9843b1072_o.png"],
            [6, "gradio_files/samples/flickr-landscape/images/8042-7987076838_05973d5ee8_o.png", "gradio_files/samples/flickr-landscape/colored/8042-7987076838_05973d5ee8_o.png"],
        ]
    
    @staticmethod
    def get_labels(input_id):
        return ["None"] + list(AppUtils._label_id_map[input_id].keys())
    
    @staticmethod
    def get_inst_id(input_id, label):
        return AppUtils._inst_id_map[input_id][label]
    
    @staticmethod
    def get_label_id(input_id, label):
        return AppUtils._label_id_map[input_id][label]

    @staticmethod
    def _create_folder(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def copy_file(src_path, dest_path):
        os.system(f"cp {src_path} {dest_path}")