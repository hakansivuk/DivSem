from PIL import Image
import numpy as np
from utils import AppUtils
import cv2
from inference import start_inference

class AppInference:
    def __init__(self):
        self.COLOR_MAP = {}    

    def inference(self, input_id, img_path, mask, label):
        AppUtils.clear()
        self._input_id = input_id
        self._handle_preprocess(img_path, mask, label)
        return self._handle_model_inference()
    
    def _handle_preprocess(self, img_path, mask, label):
        items = self._read_files(img_path)
        self._items = items
        mask = self._save_mask(items, mask)
        if label != "None":
            self._edit_maps(items, mask, label, save=True)

    def _handle_model_inference(self):
        return start_inference()
    
    def preview(self, input_id, img_path, mask, label):
        AppUtils.clear()
        self._input_id = input_id
        items = self._read_files(img_path)
        mask = self._save_mask(items, mask)
        if label != "None":
            self._edit_maps(items, mask, label)
        return self.generate_colored_image(items["inst_map"])

    def generate_colored_image(self, semantic_map):
        np.random.seed(256)
        if len(semantic_map.shape) == 3:
            semantic_map = semantic_map[:,:,1]
        color_image = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 3), dtype=np.uint8)
        for row in range(semantic_map.shape[0]):
            for col in range(semantic_map.shape[1]):
                inst_id = semantic_map[row, col]
                if self.COLOR_MAP.get(inst_id, None) is None:
                    self.COLOR_MAP[inst_id] = np.random.randint(256, size=(3,))
                color_image[row, col, :] = self.COLOR_MAP[inst_id]
        return Image.fromarray(color_image)

    def _read_files(self, img_path):
        dataset = img_path.split("/")[2]
        items = {
            "img_path": img_path,
            "label_path": img_path.replace("images", "labels").replace("jpg", "png"),
            "inst_map_path": img_path.replace("images", "inst_map").replace("jpg", "png"),
        }
        for file_path in items.values():
            AppUtils.copy_file(file_path, file_path.replace(dataset, "test_processed"))
        items["dataset"] = dataset
        base_img = cv2.imread(img_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_lab = cv2.imread(items["label_path"], 0)
        base_inst_map = Image.open(items["inst_map_path"])            
        base_inst_map = np.array(base_inst_map, dtype=np.int32)
        items.update(
            {
                "img": base_img,
                "label": base_lab,
                "inst_map": base_inst_map,
            }
        )
        return items

    def _save_mask(self, items, mask):
        mask = np.array(mask)[:,:,0]
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)
        save_path = items["img_path"].replace(items["dataset"], "test_processed").replace("images", "predefined_masks/type_0").replace("jpg", "png")
        cv2.imwrite(save_path, mask[0]* 255)
        return mask[0].astype(np.uint8)

    def _edit_maps(self, items, mask, label, save=False):
        mask_path = items["img_path"].replace(items["dataset"], "test_processed").replace("images", "predefined_masks/type_0").replace("jpg", "png")
        mask = cv2.imread(mask_path, 0) / 255
        target_pixels = mask == 1
        target_inst_id = AppUtils.get_inst_id(self._input_id, label)
        items["inst_map"][target_pixels] = target_inst_id
        items["label"][target_pixels] = (target_inst_id % 120)
        im = Image.fromarray(items["inst_map"]).convert("I")
        if save:
            im.save(items["inst_map_path"].replace(items["dataset"], "test_processed"))
            cv2.imwrite(items["label_path"].replace(items["dataset"], "test_processed"), items["label"])
