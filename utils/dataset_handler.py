from abc import ABC, abstractmethod
import pathlib
from typing import Optional


class Dataset_handler(ABC):
    """
    Abstract class for managing dataset paths.

    Args:

    Attributes:
        subjects (list[pathlib.Path]): List containing the paths of the dataset subjects.
        scope (str): Dataset scope. Can be 'trainset', 'validset', 'testset' or 'all'.
    """

    def __init__(self, ds_path: str, scope: str):
        """
        This function must return a list with the subject paths
        within the selected dataset scope.

        Args:
            scope (str): Dataset scope. Can be 'trainset', 'validset', 'testset' or 'all'.
        """
        self.ds_path = ds_path
        self.subjects = self._get_subjects()
        self.scope = scope

    @abstractmethod
    def _get_subjects(self) -> list[pathlib.Path]:
        """
        This function must return a list with the subject paths
        within the selected dataset scope.
        """
        pass

    @abstractmethod
    def get_tract_paths_from_suj(self, subject: pathlib.Path, tract: str = "all"):
        """
        This function must return a list with the paths of the tracts belonging to a subject.
        """
        pass

    @abstractmethod
    def get_anat_from_subject(self, subject: pathlib.Path):
        """
        This function must return the path of the anatomical image of a subject (T1, T2, etc.).
        """
        pass

    @abstractmethod
    def get_data_from_subject(self, subject: pathlib.Path):
        """
        This function must return a dictionary with all the information of a subject.
        """
        pass

    def __len__(self):
        """
        This function returns the number of subjects included in the selected dataset scope.
        """
        return len(self.subjects)


# =========================== TRACTOINFERNO HANDLER CLASS =========================== #
class Tractoinferno_handler(Dataset_handler):
    """
    Class for handling the Tractoinferno dataset.
    To use the Tractoinferno dataset, it must be downloaded first from
    https://openneuro.org/datasets/ds003900/versions/1.1.1
    and the path to the 'derivatives' subfolder must be provided in the
    'tractoinferno_path' parameter of the Tractoinferno class.
    """

    def __init__(self,
        tractoinferno_path: str,
        scope: str = "all"):

        # Create a pathlib Path and check that it exists
        if pathlib.Path(tractoinferno_path).exists():
            self.tractoinferno_path = pathlib.Path(tractoinferno_path)
        else:
            raise ValueError("The path to the Tractoinferno dataset does not exist.")

        # Check that the scope is valid
        if scope not in ["trainset", "validset", "testset", "all"]:
            raise ValueError("Scope must be 'trainset', 'validset', 'testset', or 'all'.")
        else:
            self.scope = scope  # Dataset scope
            self.subjects = self._get_subjects()  # List of subject paths in the dataset

        # List of tracts
        self.TRACT_LIST = {
            'AF_L': {
                'id': 0,
                'tract': 'arcuate fasciculus',
                'side': 'left',
                'type': 'association'
            }, 'AF_R': {
                'id': 1,
                'tract': 'arcuate fasciculus',
                'side': 'right',
                'type': 'association'
            }, 'CC_Fr_1': {
                'id': 2,
                'tract': 'corpus callosum, frontal lobe',
                'side': 'most anterior part of the frontal lobe',
                'type': 'commissural'
            }, 'CC_Fr_2': {
                'id': 3,
                'tract': 'corpus callosum, frontal lobe',
                'side': 'most posterior part of the frontal lobe',
                'type': 'commissural'
            }, 'CC_Oc': {
                'id': 4,
                'tract': 'corpus callosum, occipital lobe',
                'side': 'central',
                'type': 'commissural'
            }, 'CC_Pa': {
                'id': 5,
                'tract': 'corpus callosum, parietal lobe',
                'side': 'central',
                'type': 'commissural'
            }, 'CC_Pr_Po': {
                'id': 6,
                'tract': 'corpus callosum, pre/post central gyri',
                'side': 'central',
                'type': 'commissural'
            }, 'CG_L': {
                'id': 7,
                'tract': 'cingulum',
                'side': 'left',
                'type': 'association'
            }, 'CG_R': {
                'id': 8,
                'tract': 'cingulum',
                'side': 'right',
                'type': 'association'
            }, 'FAT_L': {
                'id': 9,
                'tract': 'frontal aslant tract',
                'side': 'left',
                'type': 'association'
            }, 'FAT_R': {
                'id': 10,
                'tract': 'frontal aslant tract',
                'side': 'right',
                'type': 'association'
            }, 'FPT_L': {
                'id': 11,
                'tract': 'fronto-pontine tract',
                'side': 'left',
                'type': 'association'
            }, 'FPT_R': {
                'id': 12,
                'tract': 'fronto-pontine tract',
                'side': 'right',
                'type': 'association'
            }, 'FX_L': {
                'id': 13,
                'tract': 'fornix',
                'side': 'left',
                'type': 'commissural'
            }, 'FX_R': {
                'id': 14,
                'tract': 'fornix',
                'side': 'right',
                'type': 'commissural'
            }, 'IFOF_L': {
                'id': 15,
                'tract': 'inferior fronto-occipital fasciculus',
                'side': 'left',
                'type': 'association'
            }, 'IFOF_R': {
                'id': 16,
                'tract': 'inferior fronto-occipital fasciculus',
                'side': 'right',
                'type': 'association'
            }, 'ILF_L': {
                'id': 17,
                'tract': 'inferior longitudinal fasciculus',
                'side': 'left',
                'type': 'association'
            }, 'ILF_R': {
                'id': 18,
                'tract': 'inferior longitudinal fasciculus',
                'side': 'right',
                'type': 'association'
            }, 'MCP': {
                'id': 19,
                'tract': 'middle cerebellar peduncle',
                'side': 'central',
                'type': 'commissural'
            }, 'MdLF_L': {
                'id': 20,
                'tract': 'middle longitudinal fasciculus',
                'side': 'left',
                'type': 'association'
            }, 'MdLF_R': {
                'id': 21,
                'tract': 'middle longitudinal fasciculus',
                'side': 'right',
                'type': 'association'
            }, 'OR_ML_L': {
                'id': 22,
                'tract': 'optic radiation, Meyer loop',
                'side': 'left',
                'type': 'projection'
            }, 'OR_ML_R': {
                'id': 23,
                'tract': 'optic radiation, Meyer loop',
                'side': 'right',
                'type': 'projection'
            }, 'POPT_L': {
                'id': 24,
                'tract': 'pontine crossing tract',
                'side': 'left',
                'type': 'commissural'
            }, 'POPT_R': {
                'id': 25,
                'tract': 'pontine crossing tract',
                'side': 'right',
                'type': 'commissural'
            }, 'PYT_L': {
                'id': 26,
                'tract': 'pyramidal tract',
                'side': 'left',
                'type': 'projection'
            }, 'PYT_R': {
                'id': 27,
                'tract': 'pyramidal tract',
                'side': 'right',
                'type': 'projection'
            }, 'SLF_L': {
                'id': 28,
                'tract': 'superior longitudinal fasciculus',
                'side': 'left',
                'type': 'association'
            }, 'SLF_R': {
                'id': 29,
                'tract': 'superior longitudinal fasciculus',
                'side': 'right',
                'type': 'association'
            }, 'UF_L': {
                'id': 30,
                'tract': 'uncinate fasciculus',
                'side': 'left',
                'type': 'association'
            }, 'UF_R': {
                'id': 31,
                'tract': 'uncinate fasciculus',
                'side': 'right',
                'type': 'association'
            }
        }

        # Dictionary with the same content but using the ID as the key instead of the tract name
        # Example: {'0': 'AF_L', '1': 'AF_R', ...}
        self.LABELS = {value["id"]: key for key, value in self.TRACT_LIST.items()}
        #print(self.LABELS)

    def _get_subjects(self) -> Optional[list[pathlib.Path]]:
        """
        Function to get the list of dataset subjects.

        Returns:
            list[pathlib.Path]: List of subject paths.
        """
        if self.scope == "all":
            # Return the subfolder paths from trainset, validationset, and testset folders
            return [path for path in self.tractoinferno_path.glob("*/*") if path.is_dir()]
        elif self.scope in ["trainset", "validset", "testset"]:
            # Return the subfolder paths from the selected folder
            return [path for path in self.tractoinferno_path.joinpath(self.scope).glob("*") if path.is_dir()]

    def get_tract_paths_from_suj(self,
        subject: pathlib.Path,
        tract: str = "all",
        extension: str = "trk"):
        """
        Function to get the paths of the tracts for a given subject.

        Args:
            subject (str): Subject name.
            tract (str): Tract name.

        Returns:
            list[str]: List of tract paths for the subject.
        """
        if tract not in self.TRACT_LIST.keys() and tract != "all":
            raise ValueError("The specified tract does not exist.")

        if tract == "all":
            # Return the paths of all tracts of the subject
            return [path for path in subject.joinpath("tractography").glob(f"*.{extension}")]
        else:
            # Return the path of the specified tract
            return [path for path in subject.joinpath("tractography").glob(f"*{tract}*.{extension}")]

    def get_anat_from_subject(self, subject: pathlib.Path):
        """
        Return the anatomical image (e.g., T1w) path of a subject.

        Args:
            subject (str): Subject name.

        Returns:
            pathlib.Path: Path to the subject's anatomical image.
        """
        try:
            return [path for path in subject.joinpath("anat").glob(f"*.nii.gz")][0]
        except KeyError:
            raise ValueError("The subject does not exist.")

    def get_data_from_subject(self, subject: pathlib.Path, extension: str = "trk") -> dict:
        """
        Return all data from a subject, including T1w image and tract paths.
        """
        anat = self.get_anat_from_subject(subject)  # Get the T1w image
        tracts = self.get_tract_paths_from_suj(subject, extension=extension)  # Get all tracts of the subject
        return {
            "subject": subject.name,
            "subject_split": subject.parent.name,
            "T1w": anat,
            "tracts": tracts
        }

    def get_label_from_tract(self, tract: str) -> str:
        """
        Return the label (ID) of a tract.

        Args:
            tract (str): Tract name.

        Returns:
            str: Tract label (ID).
        """
        return self.TRACT_LIST[tract]["id"]

    def get_tract_from_label(self, label: int) -> Optional[str]:
        """
        Return the tract name given its label.

        Args:
            label (int): Tract label (ID).

        Returns:
            str: Tract name.
        """
        for tract in self.TRACT_LIST.keys():
            if self.TRACT_LIST[tract]["id"] == label:
                return tract

    def get_data(self) -> list[dict]:
        """
        Return a list with the data of all subjects in the dataset.

        Output format:
        [
            { 'subject': Path, 'T1w': Path, 'tracts': [Path, Path, ...] },
            ...
        ]
        """
        if self.subjects is None:
            raise ValueError("Subjects list is None. Ensure that the dataset path and scope are correctly set.")
        return [self.get_data_from_subject(subject) for subject in self.subjects]
