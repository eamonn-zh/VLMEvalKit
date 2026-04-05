from vlmeval.dataset.video_base import VideoBaseDataset
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from ..smp.file import load
import numpy as np
import zipfile
import os


NQ_QUESTION_TYPES = [
    "object_counting_single",
    "object_counting_multiple",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation_single",
    "room_size_estimation_multiple"
]


MCQ_QUESTION_TYPES = [
    "object_rel_direction_forward_easy",
    "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard",
    "object_rel_distance_closest",
    "object_rel_distance_farthest",
    "route_planning"
]

PROMPT_PREFIX = "These are frames of a video."
REPO_ID = "3dlg-hcvc/ReVSI"


def _mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = (abs(pred - target) / target) <= (1 - conf_intervs)
    return accuracy.mean()


class ReVSI(VideoBaseDataset):

    TYPE = 'Video-QA'  # You can set an appropriate type

    def __init__(self, dataset='ReVSI', pack=False, nframe="all", **kwargs):
        self.fps = 0
        self.nframe = nframe
        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)
        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))
        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos

    @classmethod
    def supported_datasets(cls):
        return ['ReVSI']

    def prepare_dataset(self, dataset):
        subset = f"{self.nframe}_frame"
        dataset_table = load_dataset(REPO_ID, subset, split="test")
        dataset_table = dataset_table.add_column('video', [f"{x['scene_id']}.mp4" for x in dataset_table])
        df = dataset_table.to_pandas()
        video_zip_path = hf_hub_download(repo_id=REPO_ID, filename="video.zip", repo_type="dataset")
        dataset_path = os.path.dirname(video_zip_path)
        required_subdirs = ["all_frame", "16_frame", "32_frame", "64_frame"]

        if not all(
            os.path.exists(os.path.join(dataset_path, subdir)) for subdir in required_subdirs
        ):
            with zipfile.ZipFile(video_zip_path, "r") as zf:
                zf.extractall(dataset_path)
        tsv_file_path = os.path.join(dataset_path, f"{subset}.tsv")
        df.to_csv(tsv_file_path, sep="\t", index=False)
        return dict(root=dataset_path, data_file=tsv_file_path)

    def build_prompt(self, idx, video_llm):
        line = self.data.iloc[idx]
        question_type = line["question_type"]
        question = line["question"]
        if question_type in NQ_QUESTION_TYPES:
            post_prompt = "Answer the question using a single integer or decimal number."
            full_prompt = "\n".join([PROMPT_PREFIX, question, post_prompt]).strip()
        elif question_type in MCQ_QUESTION_TYPES:
            options_str = "Options:\n" + "\n".join(line["options"])
            post_prompt = "Answer with the option's letter from the given choices directly."
            full_prompt = "\n".join([PROMPT_PREFIX, question, options_str, post_prompt]).strip()
        message = []
        message.append(dict(type='video', value=os.path.join(self.data_root, f"{self.nframe}_frame", line["video"])))
        message.append(dict(type='text', value=full_prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        df = load(eval_file)
        for i, row in df.iterrows():
            pred_answer = str(row["prediction"]).strip().split(" ")[0].rstrip(".").strip()
            gt_answer = str(row["ground_truth"])
            if row["question_type"] in MCQ_QUESTION_TYPES:
                accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
            elif row["question_type"] in NQ_QUESTION_TYPES:
                try:
                    accuracy = _mean_relative_accuracy(
                        float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05
                    )
                except:
                    accuracy = 0.0
            df.at[i, "accuracy"] = accuracy

        output = {}
        for question_type, question_type_idx in df.groupby("question_type").groups.items():
            per_question_type = df.iloc[question_type_idx]
            output[f"{question_type}_accuracy"] = per_question_type["accuracy"].mean()

        rel_dir_accs = [
            output.pop("object_rel_direction_forward_easy_accuracy"),
            output.pop("object_rel_direction_backward_easy_accuracy"),
            output.pop("object_rel_direction_forward_hard_accuracy"),
            output.pop("object_rel_direction_backward_hard_accuracy"),
        ]
        output["object_rel_direction_accuracy"] = np.mean(rel_dir_accs)

        rel_dist_accs = [
            output.pop("object_rel_distance_closest_accuracy"),
            output.pop("object_rel_distance_farthest_accuracy"),
        ]
        output["object_rel_distance_accuracy"] = np.mean(rel_dist_accs)

        obj_count_accs = [
            output.pop("object_counting_single_accuracy"),
            output.pop("object_counting_multiple_accuracy"),
        ]
        output["object_counting_accuracy"] = np.mean(obj_count_accs)

        room_size_accs = [
            output.pop("room_size_estimation_single_accuracy"),
            output.pop("room_size_estimation_multiple_accuracy"),
        ]
        output["room_size_estimation_accuracy"] = np.mean(room_size_accs)
        output["overall"] = sum(output.values()) / len(output)
        return output
