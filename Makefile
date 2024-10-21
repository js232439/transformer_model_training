run:
	# docker compose up --build classification

	docker compose up --build classification
	#  Get the container ID of the classification service
	CONTAINER_ID=$(shell docker ps -qf "name=classification")
	# Copy the file from the container to the host machine
	docker cp $$CONTAINER_ID:/app/output/classified_statements.txt ./classified_statements.txt