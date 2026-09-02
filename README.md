# MPI AI Engine

## Overview

MPI AI Engine is a local AI-powered service recommendation engine for MPI.

The MVP takes a customer requirement, analyzes the request, identifies relevant MPI services, retrieves supporting MPI knowledge from a local vector database, and generates a safe response without inventing unsupported MPI facts.

The current implementation is designed to run locally using open-source components and does not require paid LLM APIs.

---

## Current Architecture

```text
Customer Requirement
        |
        v
Requirement Analyzer
        |
        v
Hybrid Service Matcher
        |
        v
MPI Knowledge Retrieval
        |
        v
Evidence Validation
        |
        v
MPI Response
        |
        v
Sources