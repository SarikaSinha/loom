# Multi Agent support requirement

## Task

The framework has single-agent and workflow orchestration, but no first-class way for one agent to hand off to another agent dynamically (as opposed to statically in a Workflow). An AgentTool — a tool whose callable is another Agent.run — would enable this pattern without a full framework redesign.
Plan for a secured and performant version of the framework supporting multi agent handling.

## What not to do
Don't plan for unnecessary general changes, restrict to adding the multi agent support only

## Goal
Create a plan document named loom/sdd/multi-agent-plan.md which will detail the steps to be carried out to create the agentic framework

## Stop Condition
Interview me relentlessly about every aspect of this plan until we reach a shared understanding.