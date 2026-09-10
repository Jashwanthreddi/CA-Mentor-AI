from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

class BlockType(str, Enum):
    HEADING="heading"
    PARAGRAPH="paragraph"
    TABLE="table"
    IMAGE="image"
    LIST="list"
    CAPTION="caption"
    FOOTNOTE="footnote"
    FORMULA="formula"
    HEADER="header"
    FOOTER="footer"

class BoundingBox(BaseModel):
    left:float
    top:float
    right:float
    bottom:float


class RawBlock(BaseModel):
    #smallest extracted doc block
    id:str
    page:int
    block_type:BlockType
    text:str=""
    bbox:BoundingBox |None=None
    confidence:float |None=None
    metadata:dict =Field(default_factory=dict)


class RawPage(BaseModel):
    page_number:int
    width:float
    height:float
    blocks:list[RawBlock] = Field(default_factory=list)


class RawDocumentMetadata(BaseModel):
    filename:str
    title:str
    pages:int
    language:str
    parser:str
    parser_version:str


class RawDocument(BaseModel):
   # raw parsed doc -> produced by extractor -> consumed by classifier

   metadata:RawDocumentMetadata
   pages:list[RawPage] = Field(default_factory=list)

   