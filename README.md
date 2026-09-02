# MPI AI Engine

## Overview

MPI AI Engine is a local AI-powered service recommendation engine designed to understand customer requirements and identify relevant MPI services.

The current MVP takes a customer requirement, analyzes the request, identifies relevant MPI services, retrieves supporting MPI knowledge from a local vector database, validates the available evidence, and produces a safe response without inventing unsupported MPI facts.

The system is designed to run locally using open-source components and does not require paid LLM APIs for runtime inference.

---

## Objective

The primary objective of the MPI AI Engine is to create an intelligent service-discovery layer for MPI.

A customer can provide a natural-language requirement such as:

```text
I am a startup and need a website, branding and GST compliance support.