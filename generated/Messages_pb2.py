# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Messages.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0eMessages.proto\x12\x06\x63ommon\"c\n\x0b\x42oundingBox\x12\x0c\n\x04ymin\x18\x01 \x01(\x05\x12\x0c\n\x04xmin\x18\x02 \x01(\x05\x12\x0c\n\x04ymax\x18\x03 \x01(\x05\x12\x0c\n\x04xmax\x18\x04 \x01(\x05\x12\r\n\x05label\x18\x05 \x01(\t\x12\r\n\x05score\x18\x06 \x01(\x02\":\n\x13ListOfBoundingBoxes\x12#\n\x06packet\x18\x01 \x03(\x0b\x32\x13.common.BoundingBox\"/\n\x0cImageRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x10\n\x08rotation\x18\x02 \x01(\x05\"\x07\n\x05\x45mptyB)\n\x12\x63om.lsorter.commonB\x13\x43ommonMessagesProtob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'Messages_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\022com.lsorter.commonB\023CommonMessagesProto'
  _BOUNDINGBOX._serialized_start=26
  _BOUNDINGBOX._serialized_end=125
  _LISTOFBOUNDINGBOXES._serialized_start=127
  _LISTOFBOUNDINGBOXES._serialized_end=185
  _IMAGEREQUEST._serialized_start=187
  _IMAGEREQUEST._serialized_end=234
  _EMPTY._serialized_start=236
  _EMPTY._serialized_end=243
# @@protoc_insertion_point(module_scope)
