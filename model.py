from functools import total_ordering
import mesa
import random
import matplotlib.pyplot as plt

# This ABM instance should ideally reflect the behaviour of the matching algorithm 
# as well as the reputation system. 
# 
# - Potentially we would have to simplify the matching algo a bit, while we can implement the rep system 1:1
# 
# - I suppose the most interesting aspect is to see whether there is some kind of dominance of a handful of processors which emerges 


class MarketPlace(mesa.Agent):

    avg_reward = 0
    total_rewards = 0 
    total_jobs_matched = 0

    jobs_to_be_matched = [] 
    advertisements = {} # TODO each advertisement should have its own unique id

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
    def register_job(self, job):
        if (not self.match_job(job)):
            self.jobs_to_be_matched.append(job)
        
    def register_advertisement(self, advertisement):
        self.advertisements[advertisement["unique_id"]] = advertisement

    def match_job(self, job): 
        def has_capacity(advertisement):
           return advertisement["capacity"] > job["resources"]
            
        matches = list(filter(has_capacity, list(self.advertisements.values()))) # TODO the matchin also has to include the pricing? 
        
        if len(matches) > 0: 
            matched_advertisement = min(matches, key=lambda x:x['price_per_cpu_second'])
            matched_processor = self.model.processors[matched_advertisement["unique_id"]]

            # remove the matched advertisement from the open advertisements
            self.advertisements.pop(matched_advertisement["unique_id"])

            matched_processor.process_job(job, self.avg_reward)

            self.total_jobs_matched += 1
            self.total_rewards += job['reward']
            self.avg_reward = self.total_rewards / self.total_jobs_matched # TODO verify

            return True
            # print(matched_processor["unique_id"])
            # upon having matched a how we need to ... 
            # - remove the advertisement
            # - execute the job via the processor
        else: 
            return False
        


    def step(self):
        for job in self.jobs_to_be_matched: 
            self.match_job(job)

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
        self.model.market_place.register_job(job) # TODO with p = 0.5 a consumer specifies a min reputation 
        

# perhaps we could later have multiple processors implementing different strategies regarding price making etc. 
class ProcessorAgent(mesa.Agent):
    has_open_advertisement = False
    income = 0 
    
    advertisement = {}
    r = 0 
    s = 0 
    reputation = 0

    def __init__(self, success_rate, unique_id, model):
        super().__init__(unique_id, model)
        self.price_per_cpu_second = random.randint(1, 10)
        self.capacity = random.randint(1, 10)
        self.success_rate = success_rate

    def update_reputation(self, job, avg_reward, is_fulfilled):
        w = job['reward'] / (job['reward'] + avg_reward)
        λ = self.model.lmbda
        threshold = (self.r - self.r*λ + self.s*λ)/(self.s+1)
        if is_fulfilled:      
            if w > threshold: 
                self.r *= λ 
                self.s *= λ 
                self.r += w
        else: 
            self.r *= λ 
            self.s *= λ    
            self.s += w   

        self.reputation = self.calculate_reputation()

    def process_job(self, job, avg_reward):
        # TODO if fulfillment succesfull, with certain p
        is_fulfilled = random.uniform(0, 1) < self.success_rate
        if is_fulfilled: 
            self.income += job['reward']
            self.update_reputation(job, avg_reward, is_fulfilled)
        else: 
            self.update_reputation(job, avg_reward, False)


    def calculate_reputation(self):
        return (self.r+1) / (self.r+self.s+2)
         
    def get_income(self): 
        return self.income

    def step(self):
        def register(): 
            self.model.market_place.register_advertisement(self.advertisement)

        # if we do have an open advertisement, then adjust pricing? 
        if self.has_open_advertisement:
            if self.advertisement["price_per_cpu_second"] > 1: 
                self.advertisement["price_per_cpu_second"] -= 1
                register()

        if not self.has_open_advertisement:
            self.advertisement = {"price_per_cpu_second": self.price_per_cpu_second, "capacity": self.capacity, "unique_id": self.unique_id}
            self.has_open_advertisement = True
            register()
        
def compute_gini(processor_wealths): 
    x = sorted(processor_wealths)
    N = len(processor_wealths)
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B



class ReputationModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, M, N):

        self.num_processors = M
        self.num_consumers = N

        self.lmbda = 0.98

        self.schedule = mesa.time.BaseScheduler(self)

        self.consumers = {}
        self.processors = {}

        success_rate = 0.95
        for i in range(self.num_processors):
            unique_id = "P_"+str(i)
            a = ProcessorAgent(success_rate, unique_id, self)
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
        # self.datacollector.collect(self)

        # TODO make sure that at each step the matching is performed again
        # perhaps the processors have to adjust their pricing 

        # - match jobs
        # - execute jobs 
        # - update reputation 


all_reputations = []
ginis = []
for i in range(100): 
    all_incomes = []
    model = ReputationModel(10, 100)
    for j in range(10):
        model.step()
        incomes = [processor.get_income() for processor in model.processors.values()]
        for income in incomes: 
            all_incomes.append(income)        
        # reputations = [processor.reputation for processor in model.processors.values()]
        # for reputation in reputations: 
        #     all_reputations.append(reputation)
        # # TODO at each step in the model a certain amount of processors and/or consumers enter the market 
    gini = compute_gini(all_incomes)
    ginis.append(gini)

fig = plt.figure(figsize = (10, 5))

plt.plot(range(0, len(ginis)), ginis)
plt.savefig('ABM_plot')

# plt.xlabel("Reward Income")
# plt.ylabel("Number of Processors")
# plt.title("Students enrolled in different courses")
# plt.hist(all_incomes, bins=range(max(all_incomes) + 1))
# plt.savefig('ABM_plot')

