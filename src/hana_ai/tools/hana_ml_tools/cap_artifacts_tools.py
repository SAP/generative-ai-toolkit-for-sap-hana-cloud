"""
This module contains the functions to generate CAP artifacts.
"""

import logging
import os
from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ml.artifacts.generators.hana import HANAGeneratorForCAP

logger = logging.getLogger(__name__)

class CAPArtifactsInput(BaseModel):
    name: str = Field(description="the name of the model in model storage. If not provided, ask the user. Do not guess.")
    version: str = Field(description="the version of the model in model storage. If not provided, ask the user. Do not guess.")
    project_name: str = Field(description="the name of the project for CAP project. If not provided, ask the user. Do not guess.")
    output_dir: str = Field(description="the output directory for CAP project. If not provided, ask the user. Do not guess.")
    namespace: Optional[str] = Field(description="the namespace for CAP project, it is optional")
    cds_gen: Optional[bool] = Field(description="whether to generate CDS files for CAP project, it is optional")
    tudf: Optional[bool] = Field(description="whether to generate table UDF for CAP project, it is optional")
    archive: Optional[bool] = Field(description="whether to archive the generated artifacts, it is optional")

class CAPArtifactsTool(BaseTool):
    """
    This tool generates CAP artifacts for a given model.
    """
    name: str = "cap_artifacts"
    """Name of the tool."""
    description: str = "To generate CAP artifacts for a given model from model storage. "
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = CAPArtifactsInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self, name: str, version: str, project_name: str, output_dir: str, namespace: Optional[str] = None,
        cds_gen: Optional[bool] = False, tudf: Optional[bool] = False, archive: Optional[bool] = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name, version)

        generator = HANAGeneratorForCAP(
            project_name=project_name,
            output_dir=output_dir,
            namespace=namespace
        )
        generator.generate_artifacts(model, cds_gen=cds_gen, tudf=tudf, archive=archive)
        return "CAP artifacts generated successfully. Root directory: " + os.path.join(generator.output_dir, generator.project_name)

    async def _run_async(
        self, name: str, version: str, project_name: str, output_dir: str, namespace: Optional[str] = None,
        cds_gen: Optional[bool] = False, tudf: Optional[bool] = False, archive: Optional[bool] = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(name, version, project_name, output_dir, namespace, cds_gen, tudf, archive, run_manager)    
