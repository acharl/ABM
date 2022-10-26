import mesa
import random
# This ABM instance should ideally reflect the behaviour of the matching algorithm 
# as well as the reputation system. 
# 
# - Potentially we would have to simplify the matching algo a bit, while we can implement the rep system 1:1
# 
# - I suppose the most interesting aspect is to see whether there is some kind of dominance of a handful of processors which emerges 



class MarketPlace(mesa.Agent):
    jobs_matched = 0 
    jobs_to_be_matched = [] 
    advertisements=[]

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
    def register_job(self, job):
        if (not self.match_job(job)):
            self.jobs_to_be_matched.append(job)
        # self.jobs.append(job)
        
    def register_advertisement(self, advertisements):
        self.advertisements.append(advertisements)

    def match_job(self, job): 
        def has_capacity(advertisement):
           return advertisement["capacity"] > job["resources"]
            
        matches = list(filter(has_capacity, self.advertisements)) # TODO the matchin also has to include the pricing? 
        
        if len(matches) > 0: 
            matched_advertisement = min(matches, key=lambda x:x['price_per_cpu_second'])
            matched_processor = self.model.processors[matched_advertisement["id"]]

            # remove the matched advertisement from the open advertisements
            self.advertisements[:] = [ad for ad in self.advertisements if ad.get('id') != matched_advertisement['id']] 
            matched_processor.update_reputation()
            self.jobs_matched += 1
            return True
            # print(matched_processor["unique_id"])
            # upon having matched a how we need to ... 
            # - remove the advertisement
            # - execute the job via the processor
        else: 
            return False
        


    def step(self):
        pass
        # match jobs



class ConsumerAgent(mesa.Agent):
    has_open_jobs = False
    job = {}
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cpu_seconds= random.randint(1, 10)
        self.reward= random.randint(1, 10)
        self.resources=random.randint(1, 10)

    def step(self):
        job = {"cpu_seconds": self.cpu_seconds, "reward": self.reward, "resources": self.resources}
        self.model.market_place.register_job(job)
        

# perhaps we could later have multiple processors implementing different strategies regarding price making etc. 
class ProcessorAgent(mesa.Agent):
    has_open_advertisement = False

    r = 0 
    s = 0 
    reputation = 0

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.price_per_cpu_second = random.randint(1, 10)
        self.capacity= random.randint(1, 10)

    def update_reputation(self):
        self.reputation += 1

    def execute_job():
        pass
         
    def step(self):
        if not self.has_open_advertisement:
            advertisement = {"price_per_cpu_second": self.price_per_cpu_second, "capacity": self.capacity, "id": self.unique_id}
            self.model.market_place.register_advertisement(advertisement)
            self.has_open_advertisement = True
        

        


class ReputationModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, M, N):

        self.num_processors = M
        self.num_consumers = N
        self.schedule = mesa.time.BaseScheduler(self)

        # Create agents

        # price_per_cpu_second_array = [random.randint(1, 10) for i in range(self.num_processors)]
        # capacity_array = [random.randint(1, 10) for i in range(self.num_processors)]

        self.consumers = {}
        self.processors = {}

        for i in range(self.num_processors):
            unique_id = "P_"+str(i)
            a = ProcessorAgent(unique_id, self)
            self.schedule.add(a)
            self.processors[unique_id] = a
            
        for i in range(self.num_consumers):
            unique_id = "C_"+str(i)
            a = ConsumerAgent("C_"+str(i), self)
            self.schedule.add(a)


        self.market_place = MarketPlace(1, self)
        self.schedule.add(self.market_place)


    def step(self):
        self.schedule.step()
        # TODO JGD make sure that at each step the matching is performed again
        # perhaps the processors have to adjust their pricing 

        # - match jobs
        # - execute jobs 
        # - update reputation 

model = ReputationModel(10, 100)

for i in range(3):
    model.step()

print(len((model.market_place.jobs_to_be_matched)))
print((model.market_place.jobs_matched))
