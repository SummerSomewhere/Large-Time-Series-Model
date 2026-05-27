# 动物识别专家系统 (ANIMAL)

class AnimalExpertSystem:
    def __init__(self):
        # 1. 定义规则库 (以产生式形式存储)
        self.rules = {
            "M1": {"name": "哺乳动物", "formula": lambda f: f['F1'] or f['F2']},
            "M4": {"name": "鸟类", "formula": lambda f: f['F3'] or (f['F4'] and f['F5'])},
            "M2": {"name": "食肉动物", "formula": lambda f: f['M1'] and (f['F6'] or (f['F7'] and f['F8'] and f['F9']))},
            "M3": {"name": "有蹄类", "formula": lambda f: f['M1'] and (f['F10'] or f['F11'])},
            "H1": {"name": "豹", "formula": lambda f: f['M2'] and f['F12'] and f['F13']},
            "H2": {"name": "虎", "formula": lambda f: f['M2'] and f['F12'] and f['F14']},
            "H3": {"name": "长颈鹿", "formula": lambda f: f['M3'] and f['F16'] and f['F15'] and f['F13']},
            "H4": {"name": "斑马", "formula": lambda f: f['M3'] and f['F14']},
            "H5": {"name": "鸵鸟", "formula": lambda f: f['M4'] and f['F16'] and f['F15'] and f['F17']},
            "H6": {"name": "企鹅", "formula": lambda f: f['M4'] and f['F19'] and f['F17'] and f['F18']},
            "H7": {"name": "信天翁", "formula": lambda f: f['M4'] and f['F20']}
        }
        self.facts = {f"F{i}": False for i in range(1, 21)}  # 基础特征
        self.intermediate = {"M1": False, "M2": False, "M3": False, "M4": False}  # 中间结论

    # --- 深度学习感知层 ---
    def deep_learning_perception(self, image):
        """
        假设调用了一个预训练的卷积神经网络 (CNN)
        返回图像中各特征 F1-F20 的置信度
        """
        confidences = cnn_model.predict(image)  # 输出 [0.1, 0.9, ...]
        threshold = 0.7
        for i, prob in enumerate(confidences):
            if prob > threshold:
                self.facts[f"F{i + 1}"] = True

    # --- 正向推理 (Forward Chaining) ---
    def forward_inference(self):
        print("启动正向推理...")
        changed = True
        while changed:
            changed = False
            # 综合事实库：基础特征 + 中间结论
            all_known = {**self.facts, **self.intermediate}
            for key, rule in self.rules.items():
                if not (self.intermediate.get(key) or key.startswith('H')):  # 如果还没推导出该结论
                    if rule["formula"](all_known):
                        if key.startswith('H'):  # 发现最终结论
                            return key
                        self.intermediate[key] = True
                        changed = True
        return None

    # --- 反向推理 (Backward Chaining) ---
    def backward_inference(self, target_h):
        print(f"尝试验证假设: {self.rules[target_h]['name']}")
        # 简化逻辑：查找该规则缺失的特征，并重新调用感知模块或询问
        # 实际操作中，这里会递归检查子规则
        missing_facts = self.get_missing_requirements(target_h)
        for fact in missing_facts:
            # 模拟“主动视觉”：让 DL 模型在特定区域高分辨率重扫
            if self.re_examine_image(fact):
                self.facts[fact] = True
                # 重新触发正向推理验证
                return self.forward_inference()
        return None

    # --- 主控循环 ---
    def run(self, image):
        # 第一步：感知
        self.deep_learning_perception(image)

        # 第二步：初步正向推理
        result = self.forward_inference()

        # 第三步：如果没结果，启动混合推理（尝试几个可能的候选目标）
        if not result:
            candidates = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
            for h in candidates:
                result = self.backward_inference(h)
                if result: break

        if result:
            print(f"识别成功：该动物是 {self.rules[result]['name']}")
        else:
            print("识别失败：证据不足。")


# 实例化并执行
system = AnimalExpertSystem()
system.run(animal_image)