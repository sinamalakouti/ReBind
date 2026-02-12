import heapq
from torch.utils.data import Dataset
from collections import deque
import random
from accelerate.logging import get_logger


logger = get_logger(__name__)


class DynamicDataset(Dataset):
    def __init__(self, image_score_pairs):
        self.lower_20 = None  # Max-heap for the lowest 20% (using _heapify_max)
        self.upper_20 = None  # Min-heap for the highest 20%
        self.middle = None  # List for the middle 60%
        # print(image_score_pairs[0])
        self.data = [
            (pair[1], idx, pair[0]) for idx, pair in enumerate(image_score_pairs)
        ]  # (score, idx, image)
        self.ratio = 4
        self.k = len(self.data) // self.ratio

        # print(self.data[0])
        self._initialize_heaps()

    def _initialize_heaps(self):
        # Sort the data by score
        sorted_data = sorted(self.data, key=lambda x: x[0])  # Sort by score

        # Calculate 20% and 80% boundaries
        total_len = len(sorted_data)
        target_size_20 = total_len // self.ratio
        target_size_80 = total_len - target_size_20

        # Split into lower 20%, middle 60%, and upper 20%
        lower_20_data = sorted_data[:target_size_20]
        middle_data = sorted_data[target_size_20:target_size_80]
        upper_20_data = sorted_data[target_size_80:]

        # Initialize `lower_20`, `middle`, and `upper_20`
        self.lower_20 = [(score, idx, image) for score, idx, image in lower_20_data]
        self.middle = deque([(score, idx, image) for score, idx, image in middle_data])
        self.upper_20 = [(score, idx, image) for score, idx, image in upper_20_data]

        # Convert `lower_20` to a max-heap and `upper_20` to a min-heap
        heapq._heapify_max(self.lower_20)  # Max-heap for the lowest 20%
        heapq.heapify(self.upper_20)  # Min-heap for the highest 20%
        self.middle = sorted(self.middle, key=lambda x: x[0])

    def add_image_score(self, image, score):
        if score < 0.3:
            return
        # Add image and score to the complete dataset
        self.data.append((score, len(self.data), image))

        # Insert into appropriate heap based on current thresholds
        if (
            not self.lower_20 or score <= self.lower_20[0][0]
        ):  # lower_20 is a max-heap now
            self.lower_20.append(
                (score, len(self.data), image)
            )  # Temporarily add to list
            heapq._heapify_max(self.lower_20)  # Convert list to max-heap
        elif not self.upper_20 or score >= self.upper_20[0][0]:
            heapq.heappush(
                self.upper_20, (score, len(self.data), image)
            )  # Regular min-heap for top 20%
        else:
            self.middle.append(
                (score, len(self.data), image)
            )  # Otherwise, add to middle

        # Re-balance heaps to maintain 20-60-20 split
        total_len = len(self.data)
        target_size_20 = total_len // self.ratio  # 20% size target

        # Adjust lower_20 if it exceeds 20% by removing the largest score in lower_20
        while len(self.lower_20) > target_size_20:
            highest_in_lower = heapq._heappop_max(
                self.lower_20
            )  # Remove max of lower 20%
            self.middle.append(highest_in_lower)

        # Adjust upper_20 if it exceeds 20% by removing the smallest score in upper_20
        while len(self.upper_20) > target_size_20:
            lowest_in_upper = heapq.heappop(self.upper_20)  # Remove min of upper 20%
            self.middle.append(lowest_in_upper)

        # Move from middle to lower_20 if lower_20 is underfilled
        while len(self.lower_20) < target_size_20 and self.middle:
            min_middle = min(self.middle, key=lambda x: x[0])
            self.middle.remove(min_middle)
            self.lower_20.append(min_middle)
            heapq._heapify_max(self.lower_20)

        # Move from middle to upper_20 if upper_20 is underfilled
        while len(self.upper_20) < target_size_20 and self.middle:
            max_in_middle = max(self.middle, key=lambda x: x[0])
            self.middle.remove(max_in_middle)
            heapq.heappush(self.upper_20, max_in_middle)

    def get_percentile_splits(self):
        # Convert heaps to sorted lists for viewing
        lower_20_sorted = sorted(
            self.lower_20, reverse=True, key=lambda x: x[0]
        )  # Sorted list for the lowest 20%
        upper_20_sorted = sorted(
            self.upper_20, key=lambda x: x[0]
        )  # Sorted list for the highest 20%
        middle_sorted = sorted(
            self.middle, key=lambda x: x[0]
        )  # Sorted list for the middle 60%
        return lower_20_sorted, middle_sorted, upper_20_sorted

    def __len__(self):
        return len(self.data)  # todo

    def get_lower_and_upper(self, seed):
        assert len(self.lower_20) == len(
            self.upper_20
        ), f"lenght lower_20 and upper_20 must match each other! but got lower_20: {self.lower_20} and upper_20: {len(self.upper_20)}"
        # k = min(k, len(self.lower_20))

        lower_20_copy = self.lower_20.copy()
        upper_20_copy = self.upper_20.copy()
        random.seed(seed)
        random.shuffle(lower_20_copy)
        random.seed(seed)
        random.shuffle(upper_20_copy)

        lower_20_copy = lower_20_copy[: self.k]
        upper_20_copy = upper_20_copy[: self.k]
        dataset = {
            "jpg_0": [],
            "jpg_1": [],
            "score_0": [],
            "score_1": [],
            "caption": [],
        }
        dataset["jpg_0"] = [pair[2] for pair in upper_20_copy]
        dataset["jpg_1"] = [pair[2] for pair in lower_20_copy]
        dataset["idx_0"] = [pair[1] for pair in upper_20_copy]
        dataset["idx_1"] = [pair[1] for pair in lower_20_copy]
        dataset["score_0"] = [pair[0] for pair in upper_20_copy]
        dataset["score_1"] = [pair[0] for pair in lower_20_copy]
        dataset["caption"] = [
            "A photo of a mouse chasing a cat" for _ in range(len(dataset["score_0"]))
        ]
        dataset["delta_score"] = [
            score_0 - score_1
            for (score_0, score_1) in zip(dataset["score_0"], dataset["score_1"])
        ]
        import numpy as np

        logger.info(
            f"Upper scores  MEAN {np.mean(dataset['score_0'])}  MIN {np.min(dataset['score_0'])}   MAX {np.max(dataset['score_0'])}"
        )
        logger.info(
            f"Bottom scores  MEAN {np.mean(dataset['score_1'])}  MIN {np.min(dataset['score_1'])}   MAX {np.max(dataset['score_1'])}"
        )

        return dataset
