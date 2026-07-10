# We decided not to overcomplicate the sample mode simulator with this component
# class OpticalDirectionalityTranslator(SampleModeComponent, BlockModeComponent, SParameterComponent):
#     """
#     It may be natural to link a bidirectional port to an input port of one component
#     and an output port of another. One downside of this approach, is that it requires
#     every simphony simulator to update this Component with the appropriate repeator like
#     functionality.
#     """
#     ports = [
#         Port(
#             name = "bidirectional",
#             type = "optical",
#             direcitonality = "bidirectional",
#         ),
#         Port(
#             name = "in",
#             type = "optical",
#             direcitonality = "input",
#         ),
#         Port(
#             name = "out",
#             type = "optical",
#             direcitonality = "output",
#         ),
#     ]
