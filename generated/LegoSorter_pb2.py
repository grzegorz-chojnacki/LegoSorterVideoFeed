# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: LegoSorter.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import generated.Messages_pb2 as Messages__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10LegoSorter.proto\x12\x06sorter\x1a\x0eMessages.proto\"F\n\x14\x42oundingBoxWithIndex\x12\r\n\x05index\x18\x01 \x01(\x05\x12\x1f\n\x02\x62\x62\x18\x02 \x01(\x0b\x32\x13.common.BoundingBox\"N\n\x1eListOfBoundingBoxesWithIndexes\x12,\n\x06packet\x18\x01 \x03(\x0b\x32\x1c.sorter.BoundingBoxWithIndex\"$\n\x13SorterConfiguration\x12\r\n\x05speed\x18\x01 \x01(\x05\x32\xbc\x02\n\nLegoSorter\x12P\n\x10processNextImage\x12\x14.common.ImageRequest\x1a&.sorter.ListOfBoundingBoxesWithIndexes\x12>\n\x10getConfiguration\x12\r.common.Empty\x1a\x1b.sorter.SorterConfiguration\x12\x41\n\x13updateConfiguration\x12\x1b.sorter.SorterConfiguration\x1a\r.common.Empty\x12,\n\x0cstartMachine\x12\r.common.Empty\x1a\r.common.Empty\x12+\n\x0bstopMachine\x12\r.common.Empty\x1a\r.common.EmptyB%\n\x12\x63om.lsorter.sorterB\x0fLegoSorterProtob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'LegoSorter_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\022com.lsorter.sorterB\017LegoSorterProto'
  _BOUNDINGBOXWITHINDEX._serialized_start=44
  _BOUNDINGBOXWITHINDEX._serialized_end=114
  _LISTOFBOUNDINGBOXESWITHINDEXES._serialized_start=116
  _LISTOFBOUNDINGBOXESWITHINDEXES._serialized_end=194
  _SORTERCONFIGURATION._serialized_start=196
  _SORTERCONFIGURATION._serialized_end=232
  _LEGOSORTER._serialized_start=235
  _LEGOSORTER._serialized_end=551
# @@protoc_insertion_point(module_scope)